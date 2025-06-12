"""Module to load PDF files and extract it's text or data."""
from dataclasses import dataclass
import difflib
import re
import math

import pandas as pd

from .parsers import PypdfParser, OcrPdfParser, PypdfPage, OcrPage
from .representations import PdfDocumentData
from .processors import remove_hyphens, replace_ligatures

@dataclass
class DifferenceState:
    """Store state when making matches between differences from the text."""

    merged: list
    added_words: list
    removed_words: list
    missing_additions: list
    missing_removals: list
    ocr_idx: int
    status_tracker: list

class PyPDFMixedLoader():
    """Loads the information of a PDF document applying multiple verifications.

    :param cache_dir: Path to the dir to be used as cache.
    :type cache_dir: str
    :param keep_cache: True to keep the cache of images and Tessearct data. Defaults to False.
    :type keep_cache: bool
    """

    def __init__(self, cache_dir: str = './.cache', keep_cache: bool=False):
        """Mixed loader that combines text and OCR information to produce text or location data."""
        self.cache_dir = cache_dir
        self.keep_cache = keep_cache

        self.document_data = PdfDocumentData()

    def load(self, pdf_path: str, parallel: bool = False):
        """Load a full PDF document."""
        text_parser = PypdfParser(pdf_path)
        ocr_parser = OcrPdfParser(pdf_path, self.cache_dir, self.keep_cache)

        if parallel:
            pypdf_pages = list(text_parser.get_pages())
            ocr_pages = list(ocr_parser.get_pages(parallel))
        else:
            pypdf_pages = text_parser.get_pages()
            ocr_pages = ocr_parser.get_pages()

        for pypdf_page, ocr_page in zip(pypdf_pages, ocr_pages):
            self.__merge_pages(pypdf_page, ocr_page)

    def load_page(self, pdf_path:str, page_num: int):
        """Load a single page of a PDF document."""
        text_parser = PypdfParser(pdf_path)
        ocr_parser = OcrPdfParser(pdf_path, self.cache_dir, self.keep_cache)

        pypdf_page = text_parser.get_page(page_num)
        ocr_page = ocr_parser.get_page(page_num)

        self.__merge_pages(pypdf_page, ocr_page, page_num)

    def get_text(self) -> str:
        """Get the text string of the PDF document.

        Headers are removed, ligatures are replaced an hypens are fixed.
        """
        if self.document_data.is_empty():
            return ''

        text = self.document_data.get_text(remove_headers=True)
        text = replace_ligatures(text)
        text = remove_hyphens(text)

        return text

    def get_page_text(self, page_num: int):
        """Get the text string of a single page of the PDF document.

        Headers are removed, ligatures are replaced an hypens are fixed.
        """
        if self.document_data.is_empty():
            return ''

        page_text = self.document_data.get_page_text(page_num, remove_headers=True)
        page_text = replace_ligatures(page_text)
        page_text = remove_hyphens(page_text)

        return page_text

    def get_document_data(self):
        """Return the relevant OCR data of the hole document."""
        return self.document_data

    def __merge_pages(self, pypdf_page: PypdfPage, ocr_page: OcrPage, page_num: int = None):
        txt_words = pypdf_page.get_words(suffix='\n')
        ocr_words = ocr_page.get_words(suffix='\n')

        differences = list(
            difflib.Differ().compare(
                list(txt_words['word']),
                list(ocr_words['word'])))
        merged, missing_additions, missing_removals = self.__process_differences(differences)

        if missing_additions and missing_removals:
            merged.extend(self.__reconcile_missing(missing_additions, missing_removals))

        df_merged_text = pd.Series(
            (word for _, word in merged),
            index=(ocr_words.iloc[idx].name for idx, _ in merged)
        )
        self.document_data.add_page(ocr_page.get_data(), df_merged_text, page_num)

    def __process_differences(self, differences):
        #merged = [] # List of: (idx, word)
        #added_words = []
        #removed_words = []
        #missing_additions = []
        #missing_removals = []
        #ocr_idx = -1
        #status_tracker = [0, 0, 0] # curr, prev, prev_prev

        state = DifferenceState(
            merged=[], # List of: (idx, word)
            added_words=[],
            removed_words=[],
            missing_additions=[],
            missing_removals=[],
            ocr_idx = -1,
            status_tracker=[0, 0, 0] # curr, prev, prev_prev
        )

        skip = False
        for i, diff in enumerate(differences):
            curr_status = self.__get_difference_type(diff)
            if skip:
                if curr_status in (0, 1): # Don't loose count of index
                    state.ocr_idx += 1
                skip = False
                continue

            if curr_status == -1:
                continue

            state.status_tracker = [curr_status, *state.status_tracker[:2]]

            if curr_status == 0:
                state.ocr_idx += 1
                self.__handle_equal(state, diff, i)
                state.removed_words = []
                state.added_words = []

            elif curr_status == 1: # Addition
                state.ocr_idx += 1
                state.added_words.append((state.ocr_idx, diff[2:-1]))

            elif curr_status == 2: # Removal
                self.__handle_removal(state, diff)

            elif curr_status == 3: # Annotation
                skip = self.__handle_annotation(state, differences, i)
                state.added_words = []
                state.removed_words = []
                state.status_tracker[0] = 0

        state.merged.extend(self.__merge_words(state.added_words, state.removed_words))
        if not state.added_words:
            if len(state.removed_words) == 1:
                state.merged[-1] = (state.merged[-1][0],
                                    state.merged[-1][1] + state.removed_words[0])
            else:
                state.missing_removals.extend(state.removed_words)
        elif not state.removed_words:
            state.missing_additions.extend(state.added_words)

        return state.merged, state.missing_additions, state.missing_removals

    def __reconcile_missing(self, additions, removals):
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

    def __handle_equal(self, state, diff, iteration):
        single_removal = False
        if state.status_tracker[1] in [1, 2]:
            state.merged.extend(self.__merge_words(state.added_words, state.removed_words))
            if not state.added_words:
                if len(state.removed_words) == 1 and iteration > 1:
                    single_removal = True
                else:
                    state.missing_removals.extend(state.removed_words)
            elif not state.removed_words:
                state.missing_additions.extend(state.added_words)
        if single_removal:
            state.merged.append((state.ocr_idx, state.removed_words[0] + ' ' + diff[2:-1]))
        else:
            state.merged.append((state.ocr_idx, diff[2:-1]))

    def __handle_removal(self, state, diff):
        if state.status_tracker[1] == 1 and state.status_tracker[2] == 2:
            state.merged.extend(self.__merge_words(state.added_words, state.removed_words))
            if not state.added_words:
                state.missing_removals.extend(state.removed_words)
            elif not state.removed_words:
                state.missing_additions.extend(state.added_words)
            state.removed_words = []
            state.added_words = []
        state.removed_words.append(diff[2:-1])

    def __handle_annotation(self, state:DifferenceState, diffs:list, i:int) -> bool:
        annotation = diffs[i][2:-1]
        match_add = re.search(r'[+]+$', annotation)
        match_rem = re.search(r'[-]+$', annotation)
        if match_add:
            return self.__handle_annotation_addition(state, match_add, i, diffs)
        if match_rem:
            return self.__handle_annotation_removal(state, match_rem, i, diffs)

        return False

    def __handle_annotation_addition(self, state:DifferenceState, match:list, i:int, diffs:list):
        if (i + 1) >= len(diffs):
            return False  # No next_diff available

        next_diff = diffs[i + 1]

        pre_pre_prev = diffs[i - 3] if i > 2 and diffs[i - 3] != '- -\n' else None

        if pre_pre_prev is not None and match.start() == 0:
            combined = pre_pre_prev[2:-1] + state.removed_words[-1]
            if len(combined) == len(state.added_words[-1][1]):
                state.merged.extend(self.__merge_words(state.added_words[:-1],
                                                       state.removed_words[:-2]))
                if not state.added_words[:-1]:
                    state.missing_removals.extend(state.removed_words[:-2])
                elif not state.removed_words[:-2]:
                    state.missing_additions.extend(state.added_words[:-1])
                state.merged.append((state.ocr_idx, combined))
                state.added_words.clear()
                state.removed_words.clear()
                return False

        state.merged.extend(self.__merge_words(state.added_words[:-1], state.removed_words[:-1]))
        if not state.added_words[:-1]:
            state.missing_removals.extend(state.removed_words[:-1])
        elif not state.removed_words[:-1]:
            state.missing_additions.extend(state.added_words[:-1])

        if next_diff[2:-1] == state.added_words[-1][1][match.start():match.end()]:
            state.merged.append((state.ocr_idx, state.added_words[-1][1]))
            state.added_words.clear()
            state.removed_words.clear()
            return True # Skip next iteration

        state.merged.append((state.ocr_idx, state.removed_words[-1]))
        state.added_words.clear()
        state.removed_words.clear()
        return False

    def __handle_annotation_removal(self, state:DifferenceState, match:list, i:int, diffs:list):
        if (i + 1) >= len(diffs):
            return False  # No next_diff available

        state.merged.extend(self.__merge_words(state.added_words, state.removed_words[:-1]))
        if not state.added_words:
            state.missing_removals.extend(state.removed_words[:-1])
        elif not state.removed_words[:-1]:
            state.missing_additions.extend(state.added_words)

        next_diff = diffs[i + 1]
        combined = state.removed_words[-1][:match.start()] + state.removed_words[-1][match.end():]

        if combined == next_diff[2:-1]:
            state.merged.append((state.ocr_idx + 1, state.removed_words[-1]))
            state.added_words.clear()
            state.removed_words.clear()
            return True

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

        merged.append((added_words[num_iter - 1][0], ''.join(removed_words)))

        return merged

class PyPDFLoader():
    """Class to load a PDF file using pypdf library.

    This class was created only for compatibility with the extraction script.
    """
    def __init__(self, file_path:str):
        """Open the document."""
        self.parser = PypdfParser(file_path)

    def get_text(self):
        """Return the full text of the PDF file."""
        return self.parser.get_text()

    def get_page_text(self, page_num: int):
        """Return the text of a single page of the PDF file."""
        page = self.parser.get_page(page_num)
        return page.get_text()

class OCRLoader():
    """Class to load a PDF file using pytesseract to extract the text through OCR."""
    def __init__(self, file_path:str, cache_dir: str='./.cache', keep_cache: bool = False):
        """Open the document."""
        self.parser = OcrPdfParser(file_path, cache_dir, keep_cache)

    def get_text(self):
        """Return the full text of the PDF file."""
        return self.parser.get_text(remove_headers=True)

    def get_page_text(self, page_num: int):
        """Return the text of a single page of the PDF file."""
        page = self.parser.get_page(page_num)
        return page.get_raw_text()

    def get_document_data(self):
        """Return the relevant OCR data of the hole document."""
        document_data = PdfDocumentData()
        for page in self.parser.get_pages():
            document_data.add_page(page.get_data())

        return document_data
