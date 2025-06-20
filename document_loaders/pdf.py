"""Module to load PDF files and extract it's text or data."""
from dataclasses import dataclass
from enum import Enum
import difflib
import re
import math

import pandas as pd

from .parsers import PypdfParser, OcrPdfParser, PypdfPage, OcrPage, PdfPlumberParser
from .representations import PdfDocumentData
from .processors import remove_hyphens, replace_ligatures

class Status(Enum):
    """Different types of status that can be present while mergin differences."""
    NA = -1
    EQUAL = 0
    ADDITION = 1
    REMOVAL = 2
    ANNOTATION_ADD_OR_REM = 3
    ANNOTATION_CHANGE_START = 4
    ANNOTATION_CHANGE_END = 5

@dataclass
class DifferenceState:
    """Store state when making matches between differences from the text."""

    merged: list
    added_words: list
    removed_words: list
    missing_additions: list[list[(int, str)]]
    missing_removals: list[list]
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

    def get_text(self, remove_headers:bool=True, boundaries:dict[str,float]=None) -> str:
        """Get the text string of the PDF document.

        Headers are removed, ligatures are replaced an hypens are fixed.
        """
        if self.document_data.is_empty():
            return ''

        text = self.document_data.get_text(remove_headers, boundaries)
        text = replace_ligatures(text)
        text = remove_hyphens(text)

        return text

    def get_page_text(self, page_num: int, remove_headers:bool=True,
                      boundaries:dict[str,float]=None) -> str:
        """Get the text string of a single page of the PDF document.

        Headers are removed, ligatures are replaced an hypens are fixed.
        """
        if self.document_data.is_empty():
            return ''

        page_text = self.document_data.get_page_text(page_num, remove_headers=remove_headers,
                                                     boundaries=boundaries)
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
        state = DifferenceState(
            merged=[], # List of: (idx, word)
            added_words=[],
            removed_words=[],
            missing_additions=[],
            missing_removals=[],
            ocr_idx = -1,
            status_tracker=[Status.EQUAL, Status.EQUAL, Status.EQUAL] # curr, prev, prev_prev
        )

        skip = False
        for i, diff in enumerate(differences):
            curr_status = self.__get_difference_status(diff)
            if skip:
                if curr_status in (Status.EQUAL, Status.ADDITION): # Don't loose count of index
                    state.ocr_idx += 1
                skip = False
                continue

            if curr_status == Status.NA:
                continue
            if (curr_status == Status.ANNOTATION_CHANGE_START and
                state.status_tracker[1] == Status.ANNOTATION_CHANGE_START):
                curr_status = Status.ANNOTATION_CHANGE_END

            state.status_tracker = [curr_status, *state.status_tracker[:2]]

            skip = self.__handle_difference(curr_status, state, differences, i)

        if (state.added_words) != (state.removed_words):
            # When a single word is removed and is a small word, just add it to the end of the file
            if len(state.removed_words) == 1 and len(state.removed_words[0]) < 4:
                state.merged[-1] = (state.merged[-1][0],
                                    state.merged[-1][1] + "\n" + state.removed_words[0])
            else:
                self.__set_as_missing(state.added_words, state.removed_words, state)
        else:
            state.merged.extend(self.__merge_words(state.added_words, state.removed_words))

        return state.merged, state.missing_additions, state.missing_removals

    def __handle_difference(self, curr_status:Status, state:DifferenceState,
                            differences:list[str], i:int):
        diff = differences[i]
        skip = False
        if curr_status == Status.EQUAL:
            state.ocr_idx += 1
            self.__handle_equal(state, diff, i)
            state.removed_words = []
            state.added_words = []

        elif curr_status == Status.ADDITION:
            state.ocr_idx += 1
            state.added_words.append((state.ocr_idx, diff[2:-1]))

        elif curr_status == Status.REMOVAL:
            self.__handle_removal(state, diff)

        elif curr_status == Status.ANNOTATION_ADD_OR_REM:
            skip = self.__handle_annotation(state, differences, i)
            state.added_words = []
            state.removed_words = []
            state.status_tracker[0] = Status.EQUAL

        elif curr_status == Status.ANNOTATION_CHANGE_END:
            self.__handle_change_annotation(state)

        return skip

    def __reconcile_missing(self, additions:list[(int, str)],
                            removals:list[str]) -> list[(int, str)]:
        reconciled = self.__reconcile_with_exact_match(self.__flatten(additions),
                                                       self.__flatten(removals))
        if reconciled:
            return reconciled

        reconciled = self.__reconcile_with_f1_score(additions, removals)

        return reconciled

    def __get_difference_status(self, difference: str) -> Status:
        status = Status.NA
        if difference.startswith('  '):
            status = Status.EQUAL
        elif difference.startswith('+ '):
            status = Status.ADDITION
        elif difference.startswith('- '):
            status = Status.REMOVAL
        elif re.search('^[?].*[+-]+$', difference):
            status = Status.ANNOTATION_ADD_OR_REM
        elif re.search('^[?].*[^^]+$', difference):
            status = Status.ANNOTATION_CHANGE_START

        return status

    def __handle_equal(self, state, diff, iteration):
        if self.__is_text_out_of_place(state):
            self.__handle_text_out_of_place(state)

        single_removal = False
        if state.status_tracker[1] in [Status.ADDITION, Status.REMOVAL]:
            if bool(state.added_words) != bool(state.removed_words):
                # If is a single removal and is not the first of the page and is a small word
                # join the removed word with next word
                if (len(state.removed_words) == 1 and iteration > 1 and
                    len(state.removed_words[0]) < 4):
                    single_removal = True
                else:
                    self.__set_as_missing(state.added_words, state.removed_words, state)
                    state.added_words = []
                    state.removed_words = []
            else:
                state.merged.extend(self.__merge_words(state.added_words, state.removed_words))

        if single_removal:
            state.merged.append((state.ocr_idx, state.removed_words[0] + ' ' + diff[2:-1]))
        else:
            state.merged.append((state.ocr_idx, diff[2:-1]))

    def __set_as_missing(self, added_words:list[(int, str)], removed_words:list[str],
                         state:DifferenceState):
        if added_words:
            state.missing_additions.append(added_words)
        if removed_words:
            state.missing_removals.append(removed_words)

    def __handle_removal(self, state, diff):
        if state.status_tracker[1] == Status.ADDITION and state.status_tracker[2] == Status.REMOVAL:
            if bool(state.added_words) != bool(state.removed_words):
                self.__set_as_missing(state.added_words, state.removed_words, state)
            else:
                state.merged.extend(self.__merge_words(state.added_words, state.removed_words))
            state.removed_words = []
            state.added_words = []
        state.removed_words.append(diff[2:-1])

    def __handle_annotation(self, state:DifferenceState, diffs:list, i:int) -> bool:
        if self.__is_text_out_of_place(state):
            self.__handle_text_out_of_place(state)
            return False

        annotation = diffs[i][2:-1]
        match_add = re.search(r'[+]+$', annotation)
        match_rem = re.search(r'[-]+$', annotation)
        if match_add:
            return self.__handle_annotation_addition(state, match_add, i, diffs)
        if match_rem:
            return self.__handle_annotation_removal(state, match_rem, i, diffs)

        return False

    def __handle_change_annotation(self, state:DifferenceState):
        if bool(state.added_words[:-1]) or bool(state.removed_words[:-1]):
            self.__set_as_missing(state.added_words[:-1], state.removed_words[:-1], state)

        state.merged.extend(self.__merge_words([state.added_words[-1]], [state.removed_words[-1]]))
        state.added_words = []
        state.removed_words = []

    def __is_text_out_of_place(self, state:DifferenceState):
        if not state.added_words or not state.removed_words:
            return False

        has_exchanged_status = state.status_tracker[1] != state.status_tracker[2]
        have_different_size = len(state.added_words) != len(state.removed_words)

        joined_additions = ''.join(map(lambda a: a[1], state.added_words[:-1]))
        joined_removals = ''.join(state.removed_words[:-1])
        has_matching_words = joined_additions == joined_removals

        additions_words = list(joined_additions+state.added_words[-1][1])
        removals_words = list(joined_removals+state.removed_words[-1])
        exact_match_score = self.__exact_match_score(additions_words, removals_words)

        return (has_exchanged_status and have_different_size and
                (not has_matching_words and exact_match_score < 0.5))

    def __handle_text_out_of_place(self, state:DifferenceState):
        if state.added_words[:-1]:
            state.missing_additions.append(state.added_words[:-1])
        if len(state.added_words) > 0:
            state.missing_additions.append([state.added_words[-1]])
        if state.removed_words[:-1]:
            state.missing_removals.append(state.removed_words[:-1])
        if len(state.removed_words) > 0:
            state.missing_removals.append([state.removed_words[-1]])

        state.added_words = []
        state.removed_words = []

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
                if not state.added_words[:-1] and state.removed_words[:-2]:
                    state.missing_removals.append(state.removed_words[:-2])
                elif not state.removed_words[:-2] and state.added_words[:-1]:
                    state.missing_additions.append(state.added_words[:-1])
                state.merged.append((state.ocr_idx, combined))
                state.added_words = []
                state.removed_words = []
                return False

        if bool(state.added_words[:-1]) != bool(state.removed_words[:-1]):
            self.__set_as_missing(state.added_words[:-1], state.removed_words[:-1], state)
        else:
            state.merged.extend(self.__merge_words(state.added_words[:-1],
                                                   state.removed_words[:-1]))

        if next_diff[2:-1] == state.added_words[-1][1][match.start():match.end()]:
            state.merged.append((state.ocr_idx, state.added_words[-1][1]))
            state.added_words = []
            state.removed_words = []
            return True # Skip next iteration

        state.merged.append((state.ocr_idx, state.removed_words[-1]))
        state.added_words = []
        state.removed_words = []
        return False

    def __handle_annotation_removal(self, state:DifferenceState, match:list, i:int, diffs:list):
        if (i + 1) >= len(diffs):
            return False  # No next_diff available

        if bool(state.added_words) != bool(state.removed_words[:-1]):
            self.__set_as_missing(state.added_words, state.removed_words[:-1], state)
        else:
            state.merged.extend(self.__merge_words(state.added_words, state.removed_words[:-1]))

        next_diff = diffs[i + 1]
        combined = state.removed_words[-1][:match.start()] + state.removed_words[-1][match.end():]
        removed = state.removed_words[-1][match.start():match.end()]

        if combined == next_diff[2:-1]:
            if removed == state.merged[-1][1]:
                state.merged.append((state.ocr_idx + 1, combined))
            else:
                state.merged.append((state.ocr_idx + 1, state.removed_words[-1]))
            return True

        if state.removed_words:
            state.missing_removals.append([state.removed_words[-1]])

        return False

    def __reconcile_with_exact_match(self, flatten_additions:list[str],
                                     flatten_removals:list[str]) -> list[(int, str)]:
        reconciled = []
        additions_str = [a[1] for a in flatten_additions]

        idx = self.__get_exact_match_index(additions_str, flatten_removals)
        if idx > -1:
            for j, f_add in enumerate(flatten_additions):
                extended_idx = (j+idx) % len(flatten_removals)
                reconciled.append((f_add[0], flatten_removals[extended_idx]))

            return reconciled

        removals_sentence = list(''.join(flatten_removals))
        additions_sentence = list(''.join(additions_str))

        sentence_idx = self.__find_index_where_sentence_match(additions_sentence, removals_sentence)

        if sentence_idx > -1:
            offset = 0
            additions_offset = 0
            for i, f_add in enumerate(flatten_additions):
                offset += len(additions_str[i])
                if offset > sentence_idx:
                    break
                reconciled.append((f_add[0], None))
                additions_offset += 1

            offset = 0
            for i in range(additions_offset, len(flatten_additions)):
                reconciled.append((
                    flatten_additions[i][0],
                    ''.join(removals_sentence[offset:offset+len(flatten_additions[i][1])])
                ))
                offset += len(flatten_additions[i][1])

        return reconciled

    def __find_index_where_sentence_match(self, additions_sentence:list[str],
                                          removals_sentence:list[str]) -> int:
        sentence_idx = -1
        for offset in range(len(additions_sentence) - len(removals_sentence) + 1):
            score = self.__exact_match_score(
                additions_sentence[offset:offset+len(removals_sentence)],
                removals_sentence)
            if score == 1.0:
                sentence_idx = offset

        return sentence_idx

    def __reconcile_with_f1_score(self, additions:list[(int, str)],
                                  removals:list[str]) -> list[(int, str)]:
        reconciled = []
        for add in additions:
            scores = []
            for rem in removals:
                scores.append(self.__simple_f1_score(list(''.join(map(lambda a: a[1], add))),
                                            list(''.join(rem))))

            if not scores:
                continue

            max_score = max(scores)
            best_f1_idx = scores.index(max_score)

            if max_score >= 0.0:
                adds = [a[1] for a in add]
                rems = removals[best_f1_idx]
                pad_idx = self.__get_exact_match_index(adds, rems, exact=False)
                if pad_idx > -1:
                    for j, a in enumerate(add):
                        extended_idx = (j + pad_idx) % len(removals[best_f1_idx])
                        reconciled.append((a[0], removals[best_f1_idx][extended_idx]))

                removals.pop(best_f1_idx)

        return reconciled

    def __get_exact_match_index(self, a:list[str], b:list[str], exact:bool=True) -> int:
        a_len = len(a)
        b_len = len(b)
        extended_b = list(b)

        for _ in range(math.ceil(a_len / b_len)):
            extended_b += b

        scores = []
        for i in range(b_len):
            b_sample = [word.lower() for word in extended_b[i:i + a_len]]
            a_sample = [item.lower() for item in a]

            score = self.__exact_match_score(b_sample, a_sample)
            scores.append(score)
            if score == 1.0:
                break

        max_score = max(scores)
        best_idx = scores.index(max_score)

        if exact and scores[best_idx] != 1.0:
            return -1

        return best_idx

    def __exact_match_score(self, a:list[str], b:list[str]) -> float:
        total = len(a)
        count = 0
        for w_a, w_b in zip(a, b):
            if w_a == w_b:
                count += 1

        return count / total

    def __simple_f1_score(self, predictions:list[str], references:list[str]) -> float:
        # Usually articles and punctuation signs are ignored but not in this case
        pred_tokens = set(pred.strip().lower() for pred in predictions)
        ref_tokens = set(ref.strip().lower() for ref in references)

        common_tokens = ref_tokens & pred_tokens
        if len(common_tokens) == 0:
            return 0

        precission = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)

        return 2 * (precission * recall) / (precission + recall)

    def __merge_words(self, added_words:list[(int,str)], removed_words:list[str],
                      extra_joiner:str='') -> list[(int, str)]:
        if not added_words or not removed_words:
            return []

        merged = []
        n_joined_additions = len(''.join(map(lambda a: a[1], added_words)))
        n_joined_removals = len(''.join(removed_words))

        # If added and joined seem to not be related, look for a good martch
        # and if not, just use added
        if abs(n_joined_additions - n_joined_removals) > 5:
            missing_added = []
            for i, a_words in enumerate(added_words):
                for j, r_words in enumerate(removed_words):
                    f1_score = self.__simple_f1_score(list(a_words[1]),
                                                      list(r_words))
                    if f1_score >= 0.25:
                        merged.append((a_words[0], r_words))
                        removed_words.pop(j)
                        break
                else:
                    missing_added.append(a_words)

            if missing_added:
                merged.extend(missing_added)
        else:
            num_iter = min(len(added_words), len(removed_words))
            for i in range(num_iter - 1):
                merged.append((added_words[i][0], removed_words[0]))
                removed_words = removed_words[1:]

            merged.append((added_words[num_iter - 1][0], extra_joiner.join(removed_words)))

        return merged

    def __flatten(self, xss:list[object]) -> list:
        return [x for xs in xss for x in xs]

class PyPDFLoader():
    """Class to load a PDF file using pypdf library.

    This class was created only for compatibility with the extraction script.
    """
    def __init__(self, file_path:str):
        """Open the document."""
        self.parser = PypdfParser(file_path)

    def get_text(self, remove_headers:bool=True, boundaries:dict[str,float]=None):
        """Return the full text of the PDF file."""
        text = self.parser.get_text(remove_headers=remove_headers, boundaries=boundaries)
        text = replace_ligatures(text)
        text = remove_hyphens(text)

        return text

    def get_page_text(self, page_num: int, remove_headers:bool=True,
                      boundaries:dict[str,float]=None):
        """Return the text of a single page of the PDF file."""
        page = self.parser.get_page(page_num)
        page_text = page.get_text(remove_headers, boundaries)
        page_text = replace_ligatures(page_text)
        page_text = remove_hyphens(page_text)

        return page_text

class OCRLoader():
    """Class to load a PDF file using pytesseract to extract the text through OCR."""
    def __init__(self, file_path:str, cache_dir: str='./.cache', keep_cache: bool = False):
        """Open the document."""
        self.parser = OcrPdfParser(file_path, cache_dir, keep_cache)

    def get_text(self, remove_headers:bool=True, boundaries:dict[str,float]=None) -> str:
        """Return the full text of the PDF file."""
        text = self.parser.get_text(remove_headers=remove_headers, boundaries=boundaries)
        text = replace_ligatures(text)
        text = remove_hyphens(text)

        return text

    def get_page_text(self, page_num: int, remove_headers:bool=True,
                      boundaries:dict[str,float]=None) -> str:
        """Return the text of a single page of the PDF file."""
        page = self.parser.get_page(page_num)
        page_text = page.get_text(remove_headers=remove_headers, boundaries=boundaries)
        page_text = replace_ligatures(page_text)
        page_text = remove_hyphens(page_text)

        return page_text

    def get_document_data(self):
        """Return the relevant OCR data of the hole document."""
        document_data = PdfDocumentData()
        for page in self.parser.get_pages():
            document_data.add_page(page.get_data())

        return document_data

class PDFPlumberLoader():
    """Class to load a PDF file using pdfplumber with custom text reconstruction."""
    def __init__(self, file_path:str, raw:bool=False):
        self.parser = PdfPlumberParser(file_path)
        self.raw = raw

    def get_text(self, remove_headers:bool=True, boundaries:dict[str,float]=None) -> str:
        """Return the full text of the PDF file."""
        if self.raw:
            text = self.parser.get_raw_text(remove_headers=remove_headers, boundaries=boundaries)
        else:
            text = self.parser.get_text(remove_headers=remove_headers, boundaries=boundaries)
        text = replace_ligatures(text)
        text = remove_hyphens(text)

        return text

    def get_page_text(self, page_num: int, remove_headers:bool=True,
                      boundaries:dict[str,float]=None) -> str:
        """Return the text of a single page of the PDF file."""
        page = self.parser.get_page(page_num)
        if self.raw:
            page_text = page.get_raw_text(remove_headers, boundaries)
        else:
            page_text = page.get_text(remove_headers, boundaries)
        page_text = replace_ligatures(page_text)
        page_text = remove_hyphens(page_text)

        return page_text

    def get_document_data(self):
        """Return the relevant OCR data of the hole document."""
        document_data = PdfDocumentData()
        for page in self.parser.get_pages():
            document_data.add_page(page.get_data())

        return document_data
