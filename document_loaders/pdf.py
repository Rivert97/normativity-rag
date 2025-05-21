import difflib
import re
import pandas as pd
import math

from .parsers import PypdfParser, OcrPdfParser, PypdfPage, OcrPage
from .representations import PdfDocumentData
from .processors import remove_hyphens, replace_ligatures

class PyPDFMixedLoader():
    """Loads the information of a PDF document applying multiple verifications.

    :param cache_dir: Path to the dir to be used as cache.
    :type cache_dir: str
    :param keep_cache: True to keep the cache of images and Tessearct data. Defaults to False.
    :type keep_cache: bool
    """

    def __init__(self, cache_dir: str = './.cache', keep_cache: bool=False):
        self.cache_dir = cache_dir
        self.keep_cache = keep_cache

        self.documentData = PdfDocumentData()

    def load(self, pdf_path: str):
        self.text_parser = PypdfParser(pdf_path)
        self.ocr_parser = OcrPdfParser(pdf_path, self.cache_dir, self.keep_cache)

        for pypdf_page, ocr_page in zip(self.text_parser.get_pages(), self.ocr_parser.get_pages()):
            self.__merge_pages(pypdf_page, ocr_page)

    def load_page(self, pdf_path:str, page_num: int):
        self.text_parser = PypdfParser(pdf_path)
        self.ocr_parser = OcrPdfParser(pdf_path, self.cache_dir, self.keep_cache)

        pypdf_page = self.text_parser.get_page(page_num)
        ocr_page = self.ocr_parser.get_page(page_num)

        self.__merge_pages(pypdf_page, ocr_page, page_num)

    def get_text(self):
        if self.documentData.is_empty():
            return ''

        text = self.documentData.get_text(remove_headers=True)
        text = replace_ligatures(text)
        text = remove_hyphens(text)

        return text

    def get_page_text(self, page_num: int):
        if self.documentData.is_empty():
            return ''

        page_text = self.documentData.get_page_text(page_num, remove_headers=True)
        page_text = replace_ligatures(page_text)
        page_text = remove_hyphens(page_text)

        return page_text

    def get_document_data(self):
        return self.documentData

    def clear_cache(self):
        self.ocr_parser.clear_cache()

    def __merge_pages(self, pypdf_page: PypdfPage, ocr_page: OcrPage, page_num: int = None):
        txt_words = pypdf_page.get_words(suffix='\n')
        ocr_words = ocr_page.get_words(suffix='\n')

        differences = list(difflib.Differ().compare(list(txt_words['word']), list(ocr_words['word'])))
        merged, missing_additions, missing_removals = self.__process_differences(differences, ocr_words)

        if missing_additions and missing_removals:
            merged.extend(self.__reconcile_missing(missing_additions, missing_removals))

        df_merged_text = pd.Series(
            (word for _, word in merged),
            index=(ocr_words.iloc[idx].name for idx, _ in merged)
        )
        self.documentData.add_page(ocr_page.get_data(), df_merged_text, page_num)

    def __process_differences(self, differences, ocr_words):
        merged = [] # List of: (idx, word)
        added_words = []
        removed_words = []
        missing_additions = []
        missing_removals = []
        ocr_idx = -1
        skip = False
        status_tracker = [0, 0, 0] # curr, prev, prev_prev

        for i, diff in enumerate(differences):
            curr_status = self.__get_difference_type(diff)
            if skip:
                if curr_status in (0, 1): # Don't loose count of index
                    ocr_idx += 1
                skip = False
                continue

            if curr_status == -1:
                continue

            status_tracker = [curr_status, *status_tracker[:2]]

            if curr_status == 0:
                ocr_idx += 1
                single_removal = False
                if status_tracker[1] in [1, 2]:
                    merged.extend(self.__merge_words(added_words, removed_words))
                    if not added_words:
                        if len(removed_words) == 1 and i > 1:
                            single_removal = True
                        else:
                            missing_removals.extend(removed_words)
                    elif not removed_words:
                        missing_additions.extend(added_words)
                if single_removal:
                    merged.append((ocr_idx, removed_words[0] + ' ' + diff[2:-1]))
                else:
                    merged.append((ocr_idx, diff[2:-1]))
                removed_words = []
                added_words = []

            elif curr_status == 1: # Addition
                ocr_idx += 1
                added_words.append((ocr_idx, diff[2:-1]))

            elif curr_status == 2: # Removal
                if status_tracker[1] == 1 and status_tracker[2] == 2:
                    merged.extend(self.__merge_words(added_words, removed_words))
                    if not added_words:
                        missing_removals.extend(removed_words)
                    elif not removed_words:
                        missing_additions.extend(added_words)
                    removed_words = []
                    added_words = []
                removed_words.append(diff[2:-1])

            elif curr_status == 3: # Annotation
                skip = self.__handle_annotation(
                    i, differences, added_words, removed_words,
                    merged, missing_additions, missing_removals, ocr_idx
                )
                added_words = []
                removed_words = []
                status_tracker[0] = 0

        merged.extend(self.__merge_words(added_words, removed_words))
        if not added_words:
            if len(removed_words) == 1:
                merged[-1] = (merged[-1][0], merged[-1][1] + removed_words[0])
            else:
                missing_removals.extend(removed_words)
        elif not removed_words:
            missing_additions.extend(added_words)

        return merged, missing_additions, missing_removals

    def __reconcile_missing(self, additions, removals):
        # TODO: Analizar casos en los que hay más adiciones que remociones
        #TODO: Esto tendia que ser por bloque
        additions_len = len(additions)
        removals_len = len(removals)
        multi_removals = list(removals)

        for _ in range(math.ceil(additions_len / removals_len)):
            multi_removals += removals

        similarities = []
        for i in range(removals_len):
            rem_sample = [word.lower() for word in multi_removals[i:i + additions_len]]
            add_sample = [item[1].lower() for item in additions]
            similarities.append(self.__calculate_similarity(rem_sample, add_sample))

        max_sim = max(similarities)
        best_idx = similarities.index(max_sim)

        if max_sim >= 0.0:
            return [(additions[i][0], multi_removals[best_idx + i]) for i in range(additions_len)]
        return []

    def __get_difference_type(self, difference: str):
        status = -1
        if difference.startswith('  '):
            status = 0
        elif difference.startswith('+ '):
            status = 1
        elif difference.startswith('- '):
            status = 2
        elif re.search('^[?].*[+-]+$', difference):
            status = 3

        return status

    def __handle_annotation(self, i, diffs, added, removed, merged, miss_add, miss_rem, ocr_idx):
        annotation = diffs[i][2:-1]
        match_add = re.search(r'[+]+$', annotation)
        match_rem = re.search(r'[-]+$', annotation)
        if match_add:
            return self.__handle_annotation_addition(match_add, i, diffs, added, removed, merged, miss_add, miss_rem, ocr_idx)
        elif match_rem:
            return self.__handle_annotation_removal(match_rem, i, diffs, added, removed, merged, miss_add, miss_rem, ocr_idx)
        else:
            return False

    def __handle_annotation_addition(self, match, i, diffs, added, removed, merged, miss_add, miss_rem, ocr_idx):
        if (i + 1) >= len(diffs):
            return False  # No next_diff available

        next_diff = diffs[i + 1]

        pre_pre_prev = diffs[i - 3] if i > 2 and diffs[i - 3] != '- -\n' else None

        if pre_pre_prev is not None and match.start() == 0:
            combined = pre_pre_prev[2:-1] + removed[-1]
            if len(combined) == len(added[-1][1]):
                merged.extend(self.__merge_words(added[:-1], removed[:-2]))
                if not added[:-1]:
                    miss_rem.extend(removed[:-2])
                elif not removed[:-2]:
                    miss_add.extend(added[:-1])
                merged.append((ocr_idx, combined))
                added.clear()
                removed.clear()
                return False

        merged.extend(self.__merge_words(added[:-1], removed[:-1]))
        if not added[:-1]:
            miss_rem.extend(removed[:-1])
        elif not removed[:-1]:
            miss_add.extend(added[:-1])

        if next_diff[2:-1] == added[-1][1][match.start():match.end()]:
            merged.append((ocr_idx, added[-1][1]))
            added.clear()
            removed.clear()
            return True # Skip next iteration
        else:
            merged.append((ocr_idx, removed[-1]))
            added.clear()
            removed.clear()
            return False

    def __handle_annotation_removal(self, match, i, diffs, added, removed, merged, miss_add, miss_rem, ocr_idx):
        if (i + 1) >= len(diffs):
            return False  # No next_diff available

        merged.extend(self.__merge_words(added, removed[:-1]))
        if not added:
            miss_rem.extend(removed[:-1])
        elif not removed[:-1]:
            miss_add.extend(added)

        next_diff = diffs[i + 1]
        combined = removed[-1][:match.start()] + removed[-1][match.end():]

        if combined == next_diff[2:-1]:
            merged.append((ocr_idx + 1, removed[-1]))
            added.clear()
            removed.clear()
            return True
        else:
            return False

    def __calculate_similarity(self, a, b):
        total = len(a)
        count = 0
        for w_a, w_b in zip(a, b):
            if w_a == w_b:
                count += 1
        return count / total

    def __merge_words(self, added_words, removed_words):
        if not added_words or not removed_words:
            return []

        merged = []
        num_iter = min(len(added_words), len(removed_words))
        for i in range(num_iter - 1):
            merged.append((added_words[i][0], removed_words[0]))
            removed_words = removed_words[1:]
        else:
            merged.append((added_words[num_iter - 1][0], ''.join(removed_words)))

        return merged

class PyPDFLoader():
    def __init__(self, file_path:str):
        self.parser = PypdfParser(file_path)

    def get_text(self):
        return self.parser.get_text()

    def get_page_text(self, page_num: int):
        page = self.parser.get_page(page_num)
        return page.get_text()

    def clear_cache(self):
        pass

class OCRLoader():
    def __init__(self, file_path:str, cache_dir: str='./.cache', keep_cache: bool = False):
        self.parser = OcrPdfParser(file_path, cache_dir, keep_cache)

    def get_text(self):
        return self.parser.get_text(remove_headers=True)

    def get_page_text(self, page_num: int):
        page = self.parser.get_page(page_num)
        return page.get_raw_text()

    def get_document_data(self):
        documentData = PdfDocumentData()
        for page in self.parser.get_pages():
            documentData.add_page(page.get_data())

        return documentData

    def clear_cache(self):
        self.parser.clear_cache()