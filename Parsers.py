from typing import List, Iterator, Dict
from pypdf._page import PageObject

import os
import glob
from pypdf import PdfReader
from PIL import Image, ImageDraw
import pytesseract
import pdf2image
import matplotlib.pyplot as plt
import pandas as pd
import hashlib
import shutil

from Visitors import PageTextVisitor

class PypdfPage():
    """This class stores the text extracted by PyPDF and provides
    methods to process it.
    """

    def __init__(self, page: PageObject):
        self.page = page

        self.visitor = PageTextVisitor()

    def get_text(self):
        self.visitor.set_boundaries(*self.page['/ArtBox'])
        text = self.page.extract_text(visitor_text=self.visitor.visitor_text)

        return self.__remove_out_of_bounds_text(text)

    def get_words(self, suffix:str = '') -> pd.DataFrame:
        df_words = pd.DataFrame([(idx, f'{w}{suffix}') for idx, w in enumerate(self.get_text().split())], columns=['txt_idx', 'word'])
        df_words.set_index('txt_idx', inplace=True)
        return df_words

    def __remove_out_of_bounds_text(self, text: str) -> str:
        clean_text = text
        for line in self.visitor.get_out_of_bounds_text():
            if clean_text.endswith(line):
                clean_text = clean_text[:-len(line)]
            elif clean_text.startswith(line):
                clean_text = clean_text[len(line):]

        return clean_text

class PypdfParser():
    """This is a class to parse PDF files to text using pypdf.

    :param file_path: The path of the file to be parsed.
    :type file_path: str
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

        self.reader = PdfReader(self.file_path)

    def get_text(self, page_separator: str = '') -> str:
        """Return the full text of the file as extracted by pypdf.

        :param page_separator: String to be added to separate each page,
            defaults to ''.
        :type page_separator: str, optional

        :return: A string of all the text from the document
        :rtype: str
        """
        return page_separator.join([page.extract_text() for page in self.reader.pages])

    def get_num_pages(self) -> int:
        return len(self.reader.pages)

    def get_pages(self) -> Iterator[PypdfPage]:
        """Read the document and return the text of each page as an Iterator.

        :return: An Iterator that yields the text of each page at a time
        :rtype: Iterator[str]
        """
        for page in self.reader.pages:
            yield PypdfPage(page)

    def get_page(self, page_num: int):
        return PypdfPage(self.reader.pages[page_num])

class DataReconstructor():

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

        self.data['right'] = self.data['left'] + self.data['width']
        self.data['bottom'] = self.data['top'] + self.data['height']

        writable_min_x, _, writable_max_x, _ = self.__get_writable_boundaries()
        self.writable_width = writable_max_x - writable_min_x
        self.writable_center = writable_min_x + self.writable_width * 0.5

    def get_reconstructed(self) -> pd.DataFrame:
        self.__assign_line_number()
        self.__assign_column_number()
        self.__assign_column_position()
        self.__assign_group_number()

        return self.data

    def __assign_line_number(self):
        self.data['line'] = pd.Series(dtype='int')
        words = self.data[self.data['level'] == 5].sort_values(by=['top'])

        current_line_num = -1
        current_minY, current_maxY = 0, 0
        for word_idx, word in words.iterrows():
            word_top = word['top']
            word_bottom = word['top'] + word['height']
            word_center = word['top'] + word['height'] * 0.5

            if current_minY < word_center < current_maxY: # Is same line
                current_minY = min(current_minY, word_top)
                current_maxY = max(current_maxY, word_bottom)
            else: # Is another line
                current_line_num += 1
                current_minY, current_maxY = word_top, word_bottom

            self.data.loc[word_idx, 'line'] = current_line_num

    def __assign_column_number(self):
        self.data['column'] = pd.Series(dtype='int')
        tolerance = self.writable_width * 0.05

        for line_number, line_words in self.data.groupby('line'):
            col_number = -1
            current_col = None

            sorted_words = line_words.sort_values(by='left')
            for idx, word in sorted_words.iterrows():
                word_left = word['left']
                word_right = word['left'] + word['width']

                if (current_col is not None
                    and word_left - current_col['maxX'] < tolerance):
                    # It is close enough to be in the same column
                    current_col['maxX'] = word_right
                else:
                    col_number += 1
                    current_col = {'maxX': word_right, 'minX': word_left}

                self.data.loc[idx, 'column'] = col_number

    def __assign_column_position(self):
        self.data['col_position'] = pd.Series(dtype='int')

        for (line, column), values in self.data.groupby(['line', 'column']).agg({'left': ['min'], 'right': ['max']}).iterrows():
            position = -1
            if self.__column_is_aligned_left(values['left', 'min'], values['right', 'max']):
                position = 0
            elif self.__column_is_aligned_right(values['left', 'min'], values['right', 'max']):
                position = 1

            words_indices = self.data[(self.data['line'] == line) & (self.data['column'] == column)]
            self.data.loc[words_indices.index, 'col_position'] = position

    def __column_is_centered(self, min_x, max_x):
        center_rate = (self.writable_center - min_x) / (max_x - self.writable_center)
        column_percentage = (max_x - min_x) / self.writable_width
        if min_x < self.writable_center < max_x and abs(1.0 - center_rate) < 0.1 and column_percentage < 0.9:
            return True
        else:
            return False

    def __column_is_aligned_left(self, min_x, max_x):
        col_center = min_x + (max_x - min_x) * 0.5

        if col_center < self.writable_center + self.writable_width * 0.1 and not min_x > self.writable_center:
            return True
        else:
            return False

    def __column_is_aligned_right(self, min_x, max_x):
        col_center = min_x + (max_x - min_x) * 0.5

        if col_center > self.writable_center:
            return True
        else:
            return False

    def __assign_group_number(self):
        self.data['group'] = pd.Series(dtype='int')
        tolerance = self.writable_width * 0.005

        prev_num_cols = None
        prev_had_centered = None
        prev_new_group = False
        group_cols = None
        group_num = 0

        for line_number, line_words in self.data.groupby('line'):
            line_cols = {}
            for column, values in line_words.groupby(['column']).agg({'col_position': ['max'], 'left': ['min'], 'right': ['max']}).iterrows():
                line_cols[column] = {
                    'position': values['col_position', 'max'],
                    'minX': values['left', 'min'],
                    'maxX': values['right', 'max'],
                }
            has_centered = False
            for _, col in line_cols.items():
                if self.__column_is_centered(col['minX'], col['maxX']):
                    has_centered = True

            new_group = self.__is_new_group(
                line_cols=line_cols,
                group_cols=group_cols,
                prev_num_cols=prev_num_cols,
                prev_had_centered=prev_had_centered,
                has_centered=has_centered,
                prev_new_group=prev_new_group,
                tolerance=tolerance
            )

            if new_group:
                group_num += 1

            self.data.loc[line_words.index, 'group'] = group_num

            # Update tracking variables
            prev_num_cols = len(line_cols)
            prev_had_centered = has_centered
            prev_new_group = new_group

            if group_cols is None or new_group:
                group_cols = line_cols
            else:
                self.__expand_group_columns(group_cols, line_cols)

    def __is_new_group(self, line_cols, group_cols, prev_num_cols, prev_had_centered, has_centered, prev_new_group, tolerance):
        if group_cols is None or prev_num_cols is None or prev_had_centered is None:
            return False

        if len(group_cols) != len(line_cols):
            if has_centered or prev_had_centered:
                return True
        elif has_centered != prev_had_centered:
            return True

        for g_col in group_cols:
            g_position = group_cols[g_col]['position']
            for l_col in line_cols:
                if line_cols[l_col]['position'] != g_position:
                    continue

                if not self.__columns_wrap_each_other(line_cols[l_col], group_cols[g_col], tolerance) and not prev_new_group:
                    return True
        return False

    def __columns_wrap_each_other(self, col1, col2, tolerance):
        return (
            (col1['minX'] > col2['minX'] - tolerance and col1['maxX'] < col2['maxX'] + tolerance) or
            (col2['minX'] > col1['minX'] - tolerance and col2['maxX'] < col1['maxX'] + tolerance)
        )

    def __expand_group_columns(self, group_cols, line_cols):
        for g_col in group_cols:
            position = group_cols[g_col]['position']
            for l_col in line_cols:
                if line_cols[l_col]['position'] != position:
                    continue

                group_cols[g_col]['minX'] = min(group_cols[g_col]['minX'], line_cols[l_col]['minX'])
                group_cols[g_col]['maxX'] = max(group_cols[g_col]['maxX'], line_cols[l_col]['maxX'])

    def __get_writable_boundaries(self):
        blocks = self.data[self.data['level'] == 2]
        min_x = blocks['left'].min()
        max_x = (blocks['left'] + blocks['width']).max()

        min_y = blocks['top'].min()
        max_y = (blocks['top'] + blocks['height']).max()

        return min_x, min_y, max_x, max_y

class OcrPage():
    """This class stores the information contained in a single page
    of a document and provides functions to process it.
    """

    def __init__(self, image: Image):
        self.image = image

        self.data = self.__get_data_from_image()
        self.boundaries = self.__get_content_boundaries()

        reconstructor = DataReconstructor(self.data)
        self.data = reconstructor.get_reconstructed()

    def get_text(self) -> str:
        return '\n'.join(self.data.sort_values(['line', 'left']).groupby(['group', 'col_position', 'line'])['text'].apply(' '.join).groupby(['group', 'col_position']).apply('\n'.join).groupby('group').apply('\n'.join))

    def get_indices(self) -> List[int]:
        return list(self.data.sort_values(['line', 'left']).reset_index().groupby(['group', 'col_position', 'line'])['index'].agg(list).groupby(['group', 'col_position']).sum().groupby('group').sum().sum())

    def get_new_text(self, remove_headers: bool = False) -> str:
        if remove_headers:
            return '\n'.join(self.data[(self.data['left'] > self.boundaries['left']) & (self.data['top'] > self.boundaries['top']) & (self.data['left'] + self.data['width'] < self.boundaries['right']) & (self.data['top'] + self.data['height'] < self.boundaries['bottom'])].dropna().sort_values(['line', 'left']).groupby(['group', 'col_position', 'line'])['new_text'].apply(' '.join).groupby(['group', 'col_position']).apply('\n'.join).groupby('group').apply('\n'.join))
        else:
            return '\n'.join(self.data.sort_values(['line', 'left']).dropna().groupby(['group', 'col_position', 'line'])['new_text'].apply(' '.join).groupby(['group', 'col_position']).apply('\n'.join).groupby('group').apply('\n'.join))

    def get_raw_text(self) -> str:
        return '\n'.join(self.data.dropna().groupby(['block_num', 'par_num', 'line_num'])['text'].apply(' '.join).groupby(['block_num', 'par_num']).apply('\n'.join).groupby('block_num').apply('\n'.join))

    def get_words(self, suffix = '') -> pd.DataFrame:
        text = self.get_text()
        if text == '':
            return pd.DataFrame(columns=['word'])

        indices = self.get_indices()
        df_words = pd.DataFrame([(idx, f'{w}{suffix}') for idx, w in zip(indices, text.split())], columns=['ocr_idx', 'word'])
        df_words.set_index('ocr_idx', inplace=True)

        return df_words

    def get_raw_words(self, suffix = '') -> List[str]:
        return [f'{w}{suffix}' for w in self.data.dropna()['text']]

    def show_detection(self, level=2):
        canvas = self.image.copy()
        draw = ImageDraw.Draw(canvas)

        # Draw boundaries
        draw.rectangle(((self.boundaries['left'], self.boundaries['top']), (self.boundaries['right'], self.boundaries['bottom'])), outline='red')

        # Draw regions
        for _, row in self.data.iterrows():
            if row['level'] == level:
                (x, y, w, h) = (row['left'], row['top'], row['width'], row['height'])
                draw.rectangle(((x, y), (x + w, y + h)), outline="green")

        plt.imshow(canvas)
        plt.show()

    def show_relevant_detection(self):
        canvas = self.image.copy()
        draw = ImageDraw.Draw(canvas)

        # Draw regions
        for group, values in self.data.dropna().groupby(['group']).agg({'left': ['min'], 'top': ['min'], 'bottom': ['max'], 'right': ['max']}).iterrows():
            (x_1, y_1, x_2, y_2) = (values['left', 'min'], values['top', 'min'], values['right', 'max'], values['bottom', 'max'])
            draw.rectangle(((x_1, y_1), (x_2, y_2)), outline="green")

        plt.imshow(canvas)
        plt.show()

    def set_new_text(self, col_data: pd.DataFrame):
        self.data['new_text'] = col_data

    def __get_data_from_image(self):
        data = pytesseract.image_to_data(self.image, lang='spa', output_type=pytesseract.Output.DATAFRAME)

        # Corregimos condición que detecta 'nan' en texto como NaN en float
        if len(data[data['text'].isna() & (data['conf'] != -1)]) > 0:
            data.loc[data['text'].isna() & (data['conf'] != -1), 'text'] = '#nan#'

        return data

    def __get_content_boundaries(self) -> Dict[str, float]:
        boundaries = {
            'left': self.data.loc[0, 'width'] * 0.05,
            'top': self.data.loc[0, 'height'] * 0.1,
            'right': self.data.loc[0, 'width'] * 0.95,
            'bottom': self.data.loc[0, 'height'] * 0.95,
        }

        return boundaries

class OcrPdfParser():
    """This class uses Google Tesseract to parse the content of PDF
    files into texto.

    :param pdf_path: Path to the PDF file to parse
    :type pdf_path: str
    :param cache_dir: Path to the directory to be used as cache.
    :type cache_dir: str
    """

    def __init__(self, pdf_path: str, cache_dir: str = './.cache'):
        self.pdf_path = pdf_path
        self.basename = os.path.basename(self.pdf_path).split('.')[0]

        self.cache_subfolder = os.path.join(cache_dir, self.basename)
        self.cache_validation_file = os.path.join(self.cache_subfolder, f'{self.basename}.md5')

        if not self.__cache_is_valid():
            self.__create_cache()
        #TODO: Guardar arreglos numpy de datos en lugar de imagenes (?)

    def get_text(self, page_separator: str = '\n') -> str:
        """Return text detected in PDF as Tesseract detects it.

        :return: String of text contained in the file
        :rtype: str
        """
        pages_path = glob.glob(f'{self.cache_subfolder}/0001-*.jpg')
        # TODO: Verificar ordenamiento
        text = ""
        for page_path in sorted(pages_path):
            data = pytesseract.image_to_data(Image.open(page_path), lang='spa', output_type=pytesseract.Output.DATAFRAME)
            text += '\n'.join(data.dropna().groupby(['block_num', 'par_num', 'line_num'])['text'].apply(' '.join).groupby(['block_num', 'par_num']).apply('\n'.join).groupby('block_num').apply('\n'.join)) + page_separator

        return text

    def get_pages(self) -> Iterator[OcrPage]:
        """Reads each page of the document and return its information as an Iterator.

        :return:
        :rtype: Iterator[str]
        """
        pages_path = glob.glob(f'{self.cache_subfolder}/0001-*.jpg')
        # TODO: Verificar ordenamiento
        for page_path in sorted(pages_path):
            yield OcrPage(Image.open(page_path))

    def get_page(self, page_num: int) -> OcrPage:
        return OcrPage(Image.open(f'{self.cache_subfolder}/0001-{page_num+1:02d}.jpg'))

    def __cache_is_valid(self) -> bool:
        if not os.path.exists(self.cache_validation_file):
            return False

        with open(self.cache_validation_file, 'r') as f:
            stored_md5 = f.read().strip()

        new_md5 = hashlib.md5(open(self.pdf_path, 'rb').read()).hexdigest()

        if stored_md5 == new_md5:
            return True
        else:
            return False

    def __create_cache(self):
        if os.path.exists(self.cache_subfolder):
            shutil.rmtree(self.cache_subfolder)
        os.makedirs(self.cache_subfolder, exist_ok=True)

        _ = pdf2image.convert_from_path(self.pdf_path, output_folder=self.cache_subfolder, fmt='jpeg', dpi=1000, output_file='')
        new_md5 = hashlib.md5(open(self.pdf_path, 'rb').read()).hexdigest()

        with open(self.cache_validation_file, 'w') as f:
            f.write(new_md5)