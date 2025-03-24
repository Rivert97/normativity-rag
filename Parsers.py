from typing import List, Iterator
from pypdf._page import PageObject

import os
import glob
from pypdf import PdfReader
from PIL import Image, ImageDraw
import pytesseract
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import pandas as pd

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

    def get_words(self, suffix = '') -> pd.DataFrame:
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
    

class OcrPage():
    """This class stores the information contained in a single page
    of a document and provides functions to process it.
    """

    def __init__(self, image: Image):
        self.image = image

        self.data = pytesseract.image_to_data(self.image, lang='spa', output_type=pytesseract.Output.DATAFRAME)
        # Corregimos condición que detecta 'nan' en texto como NaN en float
        if len(self.data[self.data['text'].isna() & (self.data['conf'] != -1)]) > 0:
            self.data.loc[self.data['text'].isna() & (self.data['conf'] != -1), 'text'] = '#nan#'
        self.boundaries = {
            'left': self.data.loc[0, 'width'] * 0.05,
            'top': self.data.loc[0, 'height'] * 0.1,
            'right': self.data.loc[0, 'width'] * 0.95,
            'bottom': self.data.loc[0, 'height'] * 0.95,
        }
        
        self.data['line'] = pd.Series(dtype='int')
        words = self.data[self.data['level'] == 5]

        writable_min_x = self.data[self.data['level'] == 2].min()['left']
        writable_max_x = (self.data[self.data['level'] == 2]['left'] + self.data[self.data['level'] == 2]['width']).max()
        writable_width = writable_max_x - writable_min_x

        current_line = None
        for word_idx, word in words.sort_values(by=['top']).iterrows():
            if current_line == None:
                current_line = {'num': 0, 'minY': word['top'], 'maxY': word['top'] + word['height']}
                self.data.loc[word_idx, 'line'] = current_line['num']
                continue
            
            w_center_y = word['top'] + (word['height'] * 0.5)
            if current_line['minY'] < w_center_y < current_line['maxY']: # Is same line
                new_minY = word['top']
                new_maxY = word['top'] + word['height']
                if new_minY < current_line['minY']:
                    current_line['minY'] = new_minY
                if new_maxY > current_line['maxY']:
                    current_line['maxY'] = new_maxY
            else: # Is another line
                current_line = {'num': current_line['num'] + 1, 'minX': word['left'], 'minY': word['top'], 'maxX': word['left'] + word['width'], 'maxY': word['top'] + word['height']}

            self.data.loc[word_idx, 'line'] = current_line['num']
        
        self.data['column'] = pd.Series(dtype='int')
        self.data['group'] = pd.Series(dtype='int')
        prev_num_cols = None
        prev_had_centered_column = None
        prev_new_group = False
        group_cols = None
        group_num = 0
        for _, line_words in self.data.groupby('line'):
            current_col = None
            cols = {
                0: None,
                1: None,
            }
            sorted_words = line_words.sort_values(by=['left'])
            for word_idx, word in sorted_words.iterrows():
                if cols[0] == None and cols[1] == None:
                    if word['left'] < writable_min_x + writable_width * 0.5:
                        num_col = 0
                    else:
                        num_col = 1
                    current_col = num_col
                    cols[current_col] = {'maxX': word['left'] + word['width'], 'minX': word['left']}
                    self.data.loc[word_idx, 'column'] = current_col
                    continue

                if word['left'] - cols[current_col]['maxX'] < writable_width*0.025:
                    cols[current_col]['maxX'] = word['left'] + word['width']
                else:
                    if word['left'] < writable_min_x + writable_width * 0.5:
                        num_col = 0
                    else:
                        num_col = 1
                    current_col = num_col
                    cols[current_col] = {'maxX': word['left'] + word['width'], 'minX': word['left']}

                self.data.loc[word_idx, 'column'] = current_col

            has_centered_column = False
            for _, value in cols.items():
                if not value:
                    continue
                if value['minX'] < writable_min_x + writable_width * 0.5 < value['maxX']:
                    has_centered_column = True
            
            new_group = False
            tolerance = writable_width * 0.005
            if group_cols != None and prev_num_cols != None and prev_had_centered_column != None:
                if prev_num_cols != current_col + 1:
                    if has_centered_column or prev_had_centered_column:
                        group_num += 1
                        new_group = True
                elif prev_new_group and group_cols[0] == None and cols[0] != None:
                    group_num += 1
                    new_group = True
                else:
                    for col in cols:
                        if cols[col] == None or group_cols[col] == None:
                            continue
                        if not(
                            (group_cols[col]['minX'] - tolerance < cols[col]['minX'] and group_cols[col]['maxX'] + tolerance > cols[col]['maxX'])
                            or (cols[col]['minX'] - tolerance < group_cols[col]['minX'] and cols[col]['maxX'] + tolerance > group_cols[col]['maxX'])
                            ) and not prev_new_group:
                            #) and (not prev_new_group and col != 1):
                            group_num += 1
                            new_group = True
                            break

            
            self.data.loc[line_words.index, 'group'] = group_num
            prev_had_centered_column = has_centered_column
            prev_num_cols = current_col + 1
            prev_new_group = new_group
            if group_cols == None or new_group:
                group_cols = cols
            else:
                for col in cols:
                    if cols[col] == None or group_cols[col] == None:
                        continue
                    if cols[col]['minX'] < group_cols[col]['minX']:
                        group_cols[col]['minX'] = cols[col]['minX']
                    if cols[col]['maxX'] > group_cols[col]['maxX']:
                        group_cols[col]['maxX'] = cols[col]['maxX']
        
    def get_text(self) -> str:
        return '\n'.join(self.data.sort_values(['line', 'left']).groupby(['group', 'column', 'line'])['text'].apply(' '.join).groupby(['group', 'column']).apply('\n'.join).groupby('group').apply('\n'.join))
    
    def get_indices(self) -> List[int]:
        return list(self.data.sort_values(['line', 'left']).reset_index().groupby(['group', 'column', 'line'])['index'].agg(list).groupby(['group', 'column']).sum().groupby('group').sum().sum())

    def get_new_text(self, remove_headers: bool = False) -> str:
        if remove_headers:
            return '\n'.join(self.data[(self.data['left'] > self.boundaries['left']) & (self.data['top'] > self.boundaries['top']) & (self.data['left'] + self.data['width'] < self.boundaries['right']) & (self.data['top'] + self.data['height'] < self.boundaries['bottom'])].dropna().sort_values(['line', 'left']).groupby(['group', 'column', 'line'])['new_text'].apply(' '.join).groupby(['group', 'column']).apply('\n'.join).groupby('group').apply('\n'.join))
        else:
            return '\n'.join(self.data.sort_values(['line', 'left']).dropna().groupby(['group', 'column', 'line'])['new_text'].apply(' '.join).groupby(['group', 'column']).apply('\n'.join).groupby('group').apply('\n'.join))

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
    
    def set_new_text(self, col_data: pd.DataFrame):
        self.data['new_text'] = col_data

class OcrPdfParser():
    """This class uses Google Tesseract to parse the content of PDF
    files into texto.

    :param pdf_path: Path to the PDF file to parse
    :type pdf_path: str
    :param cache_dir: Path to the directory to be used as cache.
    :type cache_dir: str
    """

    def __init__(self, pdf_path: str, cache_dir: str = './.cache'):
        self.basename = pdf_path.split('/')[-1].split('.')[0]

        self.cache_subfolder = f'{cache_dir}/{self.basename}'
        if not os.path.exists(self.cache_subfolder):
            os.makedirs(self.cache_subfolder)     
            #TODO: Agregar validación del caché
            _ = convert_from_path(pdf_path, output_folder=self.cache_subfolder, fmt='jpeg', dpi=1000, output_file='')
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