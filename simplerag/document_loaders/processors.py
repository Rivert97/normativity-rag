"""Module to process information from pdf parsers to apply filters or
clean the data.
"""
import re

import pandas as pd
import numpy as np
import cv2

def remove_hyphens(text: str) -> str:
    """

    This fails for:
    * Natural dashes: well-known, self-replication, use-cases, non-semantic,
                      Post-processing, Window-wise, viewpoint-dependent
    * Trailing math operands: 2 - 4
    * Names: Lopez-Ferreras, VGG-19, CIFAR-100
    """
    if not text:
        return text

    lines = [line.strip() for line in text.split("\n")]

    # Find dashes
    line_numbers_end = []
    line_numbers_start = []
    for line_no, line in enumerate(lines[:-1]):
        match_end = re.match(r'^.*[a-zA-ZñÑáéíóúÁÉÍÓÚ ]-$', line)
        if match_end:
            line_numbers_end.append(line_no)
        if line.startswith("-"):
            line_numbers_start.append(line_no)

    # Replace
    for line_no in line_numbers_end:
        lines = __dehyphenate_end(lines, line_no)
    for line_no in line_numbers_start:
        lines = __dehyphenate_start(lines, line_no)

    return "\n".join(lines)

def replace_ligatures(text: str) -> str:
    """Replace special characters present in PDF files (ligatures) with the
    corresponding utf-8 characters."""
    if not text:
        return text

    ligatures = {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ft",
        "ﬆ": "st",
        # "Ꜳ": "AA",
        # "Æ": "AE",
        "ꜳ": "aa",
    }
    for search, replace in ligatures.items():
        text = text.replace(search, replace)

    return text

def get_data_inside_boundaries(data: pd.DataFrame, boundaries:dict[str,float]=None):
    """Filter the data and return only the data that is inside the boundaries."""
    if not boundaries:
        boundaries = {
            'left': 0.05,
            'top': 0.1,
            'right': 0.95,
            'bottom': 0.95,
        }
    return data[
        (data['left'] > boundaries['left']) &
        (data['bottom'] > boundaries['top']) &
        (data['right'] < boundaries['right']) &
        (data['top'] < boundaries['bottom'])]

def __dehyphenate_end(lines: list[str], line_no: int) -> list[str]:
    next_line = lines[line_no + 1]
    word_suffix = next_line.split(" ")[0]

    if lines[line_no].endswith(" -"):
        lines[line_no] = lines[line_no][:-2] + word_suffix
    else:
        lines[line_no] = lines[line_no][:-1] + word_suffix
    lines[line_no + 1] = lines[line_no + 1][len(word_suffix) + 1:]
    return lines

def __dehyphenate_start(lines: list[str], line_no: int) -> list[str]:
    if lines[line_no] == '':
        return lines

    if lines[line_no].startswith("- "):
        word_suffix = lines[line_no].split(" ")[1]
        suffix_offset = 3
    else:
        word_suffix = lines[line_no].split(" ")[0].split("-")[1]
        suffix_offset = 1

    lines[line_no] = lines[line_no][len(word_suffix) + suffix_offset:]
    lines[line_no - 1] = lines[line_no - 1] + word_suffix
    return lines

def get_lines_from_image(image:np.array, words_data:pd.DataFrame) -> dict[str, np.array]:
    """Applies image processing techniques to extract horizontal
    and vertical lines from the image of a page of a PDF file.
    """
    lines = {
        'horizontal': np.array([], dtype=int),
        'vertical': np.array([], dtype=int),
    }

    no_words_img = __erase_words_from_image(image, words_data)

    lines['horizontal'] = __get_horizontal_lines_from_image(no_words_img)

    return lines

def __erase_words_from_image(image:np.array, words_data:pd.DataFrame) -> np.array:
    no_words_img = image.copy()

    for _, word in words_data.iterrows():
        rectangle = (
            (word['left'], word['top']),
            (word['left']+word['width'], word['top']+word['height'])
        )

        no_words_img = cv2.rectangle(no_words_img, rectangle[0], rectangle[1], 255, -1)

    return no_words_img

def __get_horizontal_lines_from_image(image:np.array) -> np.array:
    _, binary_img = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                     (int(image.shape[1]*0.02), 1))

    binary_img = cv2.erode(binary_img, horizontal_structure, iterations=1)
    binary_img = cv2.dilate(binary_img, horizontal_structure, iterations=1)

    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    h_lines = []
    for cnt in contours:
        m = cv2.moments(cnt)
        if m['m00'] == 0:
            continue

        cy = int(m['m01']/m['m00'])
        left = int(cnt[:, 0, 0].min())
        right = int(cnt[:, 0, 0].max())
        top = int(cnt[:, 0, 1].min())
        bottom = int(cnt[:, 0, 1].max())
        if bottom - top < image.shape[0]*0.001:
            h_lines.append(((left, cy), (right, cy)))

    return np.array(h_lines, dtype=int)
