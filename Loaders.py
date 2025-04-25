from typing import Dict
import difflib
import re
import pandas as pd
import math

from Parsers import PypdfParser, OcrPdfParser, PypdfPage, OcrPage
import PostProcessors

class PdfDocumentData():

    columns = ['page', 'text', 'left', 'top', 'right', 'bottom', 'line', 'column', 'col_position', 'group']

    def __init__(self):
        self.data = pd.DataFrame()

        self.page_meta = pd.DataFrame(columns=['width', 'height'])
        self.page_meta['width'] = self.page_meta['width'].astype(float)
        self.page_meta['height'] = self.page_meta['height'].astype(float)
        self.page_counter = 0

    def add_page(self, data: pd.DataFrame, merged_text: pd.DataFrame=None, page_num: int=None):
        new_data = data.copy()

        if page_num is None:
            page = self.page_counter
        else:
            page = page_num
        new_data['page'] = page

        if merged_text is not None:
            new_data['text'] = merged_text

        self.data = pd.concat([self.data, new_data.dropna()[PdfDocumentData.columns]])
        self.page_meta = pd.concat([self.page_meta, pd.DataFrame({'width': new_data.loc[0, 'width'], 'height': new_data.loc[0, 'height']}, index=[page])])

        self.page_counter += 1

    def get_last_page_text(self, remove_headers: bool=False):
        return self.get_page_text(self.page_counter - 1, remove_headers)

    def get_page_text(self, page_num: int, remove_headers: bool=False) -> str:
        if remove_headers:
            boundaries = self.__calculate_content_boundaries(page_num)
            return '\n'.join(self.data[(self.data['page'] == page_num) & (self.data['left'] > boundaries['left']) & (self.data['top'] > boundaries['top']) & (self.data['right'] < boundaries['right']) & (self.data['bottom'] < boundaries['bottom'])].dropna().sort_values(['line', 'left']).groupby(['group', 'col_position', 'line'])['text'].apply(' '.join).groupby(['group', 'col_position']).apply('\n'.join).groupby('group').apply('\n'.join))
        else:
            return '\n'.join(self.data[self.data['page'] == page_num].sort_values(['line', 'left']).dropna().groupby(['group', 'col_position', 'line'])['text'].apply(' '.join).groupby(['group', 'col_position']).apply('\n'.join).groupby('group').apply('\n'.join))

    def save_data(self, prefix: str):
        self.data.to_csv(f'{prefix}_data.csv', index=False)
        self.page_meta.to_csv(f'{prefix}_page_meta.csv')

    def __calculate_content_boundaries(self, page_num: int) -> Dict[str, float]:
        boundaries = {
            'left': self.page_meta.loc[page_num, 'width'] * 0.05,
            'top': self.page_meta.loc[page_num, 'height'] * 0.1,
            'right': self.page_meta.loc[page_num, 'width'] * 0.95,
            'bottom': self.page_meta.loc[page_num, 'height'] * 0.95,
        }

        return boundaries


class PdfMixedLoader():
    """Loads the information of a PDF document applying multiple verifications.

    :param pdf_path: Path to PDF file
    :type pdf_path: str
    :param cache_dir: Path to the dir to be used as cache.
    :type cache_dir: str
    """

    def __init__(self, pdf_path: str, cache_dir: str = './.cache', verbose:bool=False):
        self.text_parser = PypdfParser(pdf_path)
        self.ocr_parser = OcrPdfParser(pdf_path, cache_dir)
        self.verbose = verbose

        self.documentData = PdfDocumentData()

    def get_text(self):
        text = ""
        for pypdf_page, ocr_page in zip(self.text_parser.get_pages(), self.ocr_parser.get_pages()):
            self.__merge_pages(pypdf_page, ocr_page)
            page_text = self.documentData.get_last_page_text(remove_headers=True)

            if self.verbose:
                print(page_text)

            text += page_text + "\n"

        text = PostProcessors.replace_ligatures(text)
        text = PostProcessors.remove_hyphens(text)

        return text

    def get_page_text(self, page_num: int):
        pypdf_page = self.text_parser.get_page(page_num)
        ocr_page = self.ocr_parser.get_page(page_num)

        self.__merge_pages(pypdf_page, ocr_page, page_num)
        page_text = self.documentData.get_page_text(page_num, remove_headers=True)
        page_text = PostProcessors.replace_ligatures(page_text)
        page_text = PostProcessors.remove_hyphens(page_text)

        if self.verbose:
            print(page_text)

        return page_text

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
