import difflib
import re
import pandas as pd
import math

from Parsers import PypdfParser, OcrPdfParser, PypdfPage, OcrPage
import PostProcessors

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

    def get_text(self):
        text = ""
        for pypdf_page, ocr_page in zip(self.text_parser.get_pages(), self.ocr_parser.get_pages()):
            page_text = self.merge_pages(pypdf_page, ocr_page, remove_headers=True)

            if self.verbose:
                print(page_text)

            text += page_text + "\n"

        text = PostProcessors.replace_ligatures(text)
        text = PostProcessors.remove_hyphens(text)

        return text

    def get_page_text(self, page_num: int):
        pypdf_page = self.text_parser.get_page(page_num)
        ocr_page = self.ocr_parser.get_page(page_num)

        page_text = self.__merge_pages(pypdf_page, ocr_page, remove_headers=True)
        page_text = PostProcessors.replace_ligatures(page_text)
        page_text = PostProcessors.remove_hyphens(page_text)

        if self.verbose:
            print(page_text)

        return page_text

    def __merge_pages(self, pypdf_page: PypdfPage, ocr_page: OcrPage, remove_headers: bool = False):
        txt_words = pypdf_page.get_words(suffix='\n')
        ocr_words = ocr_page.get_words(suffix='\n')

        merged = [] # List of: (idx, word)
        d = difflib.Differ()
        differences = list(d.compare(list(txt_words['word']), list(ocr_words['word'])))
        curr_status = 0 # 0->Normal, 1->Adding, 2->Removing, 3->Explanation
        prev_status = 0 # 0->Normal, 1->Adding, 2->Removing, 3->Explanation
        prev_prev_status = 0
        added_words = []
        removed_words = []
        missing_additions = []
        missing_removals = []
        ocr_idx = -1
        skip_iter = False
        for i in range(len(differences)):
            #print(differences[i])
            #import pdb; pdb.set_trace()
            if skip_iter:
                skip_iter = False
                continue

            curr_diff = differences[i]

            if curr_diff.startswith('  '):
                curr_status = 0
                ocr_idx += 1
            elif curr_diff.startswith('+ '):
                curr_status = 1
                ocr_idx += 1
            elif curr_diff.startswith('- '):
                curr_status = 2
            elif re.search('^[?].*([+]+$|[^^])\n', curr_diff):
                curr_status = 3
            else:
                continue

            single_removal = False
            if curr_status == 0:
                if prev_status == 1 or prev_status == 2:
                    merged.extend(self.merge_words(added_words, removed_words))
                    if not added_words:
                        if len(removed_words) == 1 and i > 1:
                            single_removal = True
                        else:
                            missing_removals.extend(removed_words)
                    elif not removed_words:
                        missing_additions.extend(added_words)
                if single_removal:
                    merged.append((ocr_idx, removed_words[0] + ' ' + curr_diff[2:-1]))
                else:
                    merged.append((ocr_idx, curr_diff[2:-1]))
                removed_words = []
                added_words = []
            elif curr_status == 1:
                added_words.append((ocr_idx, curr_diff[2:-1]))
            elif curr_status == 2:
                if prev_status == 1 and prev_prev_status == 2:
                    merged.extend(self.merge_words(added_words, removed_words))
                    if not added_words:
                        missing_removals.extend(removed_words)
                    elif not removed_words:
                        missing_additions.extend(added_words)
                    removed_words = []
                    added_words = []
                removed_words.append(curr_diff[2:-1])
            elif curr_status == 3:
                annotation = curr_diff[2:-1]
                match = re.search('[+]+$', annotation)
                if match:
                    if (i + 1) < len(differences):
                        # TODO: Mejorar la estrategia para comparar cambios con el ? y los +
                        next_diff = differences[i + 1]
                        if i > 2 and differences[i - 3] != '- -\n':
                            pre_pre_prev_diff = differences[i - 3]
                        else:
                            pre_pre_prev_diff = None
                        #if match.start() == 0 and self.compare_words(pre_pre_prev_diff[2:-1], added_words[-1][1][match.start():match.end()]) > 0.0:
                        if pre_pre_prev_diff != None and match.start() == 0 and len(pre_pre_prev_diff[2:-1] + removed_words[-1]) == len(added_words[-1][1]):
                            merged.extend(self.merge_words(added_words[:-1], removed_words[:-2]))
                            if not added_words[:-1]:
                                missing_removals.extend(removed_words[:-2])
                            elif not removed_words[:-2]:
                                missing_additions.extend(added_words[:-1])
                            merged.append((ocr_idx, pre_pre_prev_diff[2:-1] + removed_words[-1]))
                        else:
                            merged.extend(self.merge_words(added_words[:-1], removed_words[:-1]))
                            if not added_words[:-1]:
                                missing_removals.extend(removed_words[:-1])
                            elif not removed_words[:-1]:
                                missing_additions.extend(added_words[:-1])
                            if next_diff[2:-1] == added_words[-1][1][match.start():match.end()]:
                                merged.append((ocr_idx, added_words[-1][1]))
                                skip_iter = True
                            else:
                                merged.append((ocr_idx, removed_words[-1]))
                        added_words = []
                        removed_words = []
                        curr_status = 0
                else:
                    if not added_words:
                        missing_removals.extend(removed_words)
                    elif not removed_words[:-1]:
                        missing_additions.extend(added_words)
                    else:
                        merged.extend(self.merge_words(added_words, removed_words))
                    added_words = []
                    removed_words = []
                    curr_status = 0

            prev_prev_status = prev_status
            prev_status = curr_status
        else:
            merged.extend(self.merge_words(added_words, removed_words))
            if not added_words:
                if len(removed_words) == 1:
                    merged[-1] = (merged[-1][0], merged[-1][1] + removed_words[0])
                else:
                    missing_removals.extend(removed_words)
            elif not removed_words:
                missing_additions.extend(added_words)

        # TODO: Analizar casos en los que hay más adiciones que remociones
        #missing_removals = list(filter(lambda a: a != '-', missing_removals))
        if missing_additions and missing_removals:
            #TODO: Esto tendia que ser por bloque
            additions_len = len(missing_additions)
            removals_len = len(missing_removals)
            similarities = []
            multi_removals = list(missing_removals)
            for i in range(math.ceil(additions_len/removals_len)):
                multi_removals += missing_removals
            for i in range(removals_len):
                removals_sample = [word.lower() for word in multi_removals[i:i+additions_len]]
                additions_sample = [item[1].lower() for item in missing_additions]
                similarity = self.jaccard_similarity_V2(removals_sample, additions_sample)
                similarities.append(similarity)
            #similarities = []
            #for i in range(removals_len - additions_len + 1):
            #    removals_sample = [word.lower() for word in missing_removals[i:i+additions_len]]
            #    additions_sample = [item[1].lower() for item in missing_additions]
            #    similarities.append(self.jaccard_similarity(removals_sample, additions_sample))
            max_similarity = max(similarities)
            max_similarity_idx = similarities.index(max_similarity)
            if max_similarity >= 0.0:
                for i in range(additions_len):
                    missing_additions[i] = (missing_additions[i][0], multi_removals[max_similarity_idx + i])
                    #missing_additions[i] = (missing_additions[i][0], missing_removals[max_similarity_idx + i])
            else:
                missing_additions = []

            merged.extend(missing_additions)

        df_merged = pd.Series(map(lambda x: x[1], merged), index=map(lambda x: ocr_words.iloc[x[0]].name, merged))
        ocr_page.set_new_text(df_merged)

        return ocr_page.get_new_text(remove_headers)

    def jaccard_similarity_V2(self, a, b):
        total = len(a)
        count = 0
        for w_a, w_b in zip(a, b):
            if w_a == w_b:
                count += 1
        return count / total

    def jaccard_similarity(self, a, b):
        a = set(a)
        b = set(b)
        return len(a.intersection(b)) / len(a.union(b))

    def merge_words(self, added_words, removed_words):
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

    def compare_words(self, w1, w2):
        if len(w1) != len(w2):
            return 0

        matches = 0
        count = 0
        for l1, l2 in zip(w1, w2):
            if l1.lower() == l2.lower():
                matches += 1
            count += 1

        return matches/count
