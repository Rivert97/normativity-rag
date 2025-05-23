"""Module to provide classes to store data from PDF files read by Tesseract."""
import pandas as pd

class PdfDocumentData():
    """Class to store relevant data of the OCR text."""

    columns = ['page', 'text', 'left', 'top', 'right', 'bottom',
               'line', 'column', 'col_position', 'group']

    def __init__(self):
        self.data = pd.DataFrame()

        self.page_counter = 0
        self.line_counter = 0
        self.group_counter = 0
        self.boundaries = {
            'left': 0.05,
            'top': 0.1,
            'right': 0.95,
            'bottom': 0.95,
        }

    def add_page(self, data: pd.DataFrame, merged_text: pd.DataFrame=None, page_num: int=None):
        """Append a new page to the document.

        Recalculate groups and lines to avoid duplications.
        """
        new_data = data.dropna().copy()

        if new_data.empty:
            self.page_counter += 1
            return

        if page_num is None:
            page = self.page_counter
        else:
            page = page_num
        new_data.loc[:, 'page'] = page

        if merged_text is not None:
            new_data.loc[:, 'text'] = merged_text

        new_data.loc[:, 'line'] += self.line_counter
        self.line_counter = new_data.loc[:, 'line'].max() + 1

        new_data.loc[:, 'group'] += self.group_counter
        self.group_counter = new_data.loc[:, 'group'].max() + 1

        self.data = pd.concat([self.data, new_data[PdfDocumentData.columns]], ignore_index=True)

        self.page_counter += 1

    def get_text(self, remove_headers: bool=False):
        """Get text from all the document."""
        if remove_headers:
            data = self.data[
                (self.data['left'] > self.boundaries['left']) &
                (self.data['top'] > self.boundaries['top']) &
                (self.data['right'] < self.boundaries['right']) &
                (self.data['bottom'] < self.boundaries['bottom'])]
        else:
            data = self.data

        data = data.dropna().sort_values(['page', 'line', 'left'])
        texts_by_line = data.groupby([
            'page', 'group', 'col_position', 'line'])['text'].apply(' '.join)
        texts_by_column = texts_by_line.groupby(['page', 'group', 'col_position']).apply('\n'.join)
        texts_by_group = texts_by_column.groupby(['page', 'group']).apply('\n'.join)
        texts_by_page = texts_by_group.groupby('page').apply('\n'.join)

        return '\n'.join(texts_by_page)

    def get_last_page_text(self, remove_headers: bool=False):
        """Get text of the last page added to the document."""
        return self.get_page_text(self.page_counter - 1, remove_headers)

    def get_page_text(self, page_num: int, remove_headers: bool=False) -> str:
        """Get text of certain page of the document."""
        if remove_headers:
            data = self.data[
                (self.data['page'] == page_num) &
                (self.data['left'] > self.boundaries['left']) &
                (self.data['top'] > self.boundaries['top']) &
                (self.data['right'] < self.boundaries['right']) &
                (self.data['bottom'] < self.boundaries['bottom'])]
        else:
            data = self.data[self.data['page'] == page_num]

        data = data.sort_values(['line', 'left']).dropna()
        texts_by_line = data.groupby(['group', 'col_position', 'line'])['text'].apply(' '.join)
        texts_by_column = texts_by_line.groupby(['group', 'col_position']).apply('\n'.join)
        texts_by_group = texts_by_column.groupby('group').apply('\n'.join)

        return '\n'.join(texts_by_group)

    def get_data(self, remove_headers: bool=False):
        """Get data of all the documet."""
        if remove_headers:
            return self.data[
                (self.data['left'] > self.boundaries['left']) &
                (self.data['top'] > self.boundaries['top']) &
                (self.data['right'] < self.boundaries['right']) &
                (self.data['bottom'] < self.boundaries['bottom'])]

        return self.data

    def get_page_data(self, page_num: int, remove_headers: bool=False):
        """Get data of certain page of the document."""
        if remove_headers:
            return self.data[
                (self.data['page'] == page_num) &
                (self.data['left'] > self.boundaries['left']) &
                (self.data['top'] > self.boundaries['top']) &
                (self.data['right'] < self.boundaries['right']) &
                (self.data['bottom'] < self.boundaries['bottom'])]

        return self.data[self.data['page'] == page_num]

    def save_data(self, filename: str):
        """Save all the data of the document in a CSV file."""
        self.data.to_csv(filename, index=False)

    def save_page_data(self, page_num: int, filename: str):
        """Save the data of certain page of the document in a CSV file."""
        self.data[self.data['page'] == page_num].to_csv(filename, index=False)

    def load_data(self, filename: str):
        """Load the data from a CSV file."""
        self.data = pd.read_csv(filename, sep=',').dropna()

    def is_empty(self):
        """True if no pages had been added to the document."""
        return self.page_counter == 0
