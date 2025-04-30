import pandas as pd

class PdfDocumentData():

    columns = ['page', 'text', 'left', 'top', 'right', 'bottom', 'line', 'column', 'col_position', 'group']

    def __init__(self):
        self.data = pd.DataFrame()

        self.page_counter = 0
        self.boundaries = {
            'left': 0.05,
            'top': 0.1,
            'right': 0.95,
            'bottom': 0.95,
        }

    def add_page(self, data: pd.DataFrame, merged_text: pd.DataFrame=None, page_num: int=None):
        new_data = data.copy()

        if page_num is None:
            page = self.page_counter
        else:
            page = page_num
        new_data['page'] = page

        if merged_text is not None:
            new_data['text'] = merged_text

        self.data = pd.concat([self.data, new_data.dropna()[PdfDocumentData.columns]], ignore_index=True)

        self.page_counter += 1

    def get_text(self, remove_headers: bool=False):
        if remove_headers:
            return '\n'.join(self.data[(self.data['left'] > self.boundaries['left']) & (self.data['top'] > self.boundaries['top']) & (self.data['right'] < self.boundaries['right']) & (self.data['bottom'] < self.boundaries['bottom'])].dropna().sort_values(['page', 'line', 'left']).groupby(['page', 'group', 'col_position', 'line'])['text'].apply(' '.join).groupby(['page', 'group', 'col_position']).apply('\n'.join).groupby(['page', 'group']).apply('\n'.join).groupby('page').apply('\n'.join))
        else:
            return '\n'.join(self.data.sort_values(['page', 'line', 'left']).dropna().groupby(['page', 'group', 'col_position', 'line'])['text'].apply(' '.join).groupby(['page', 'group', 'col_position']).apply('\n'.join).groupby(['page', 'group']).apply('\n'.join).groupby('page').apply('\n'.join))

    def get_last_page_text(self, remove_headers: bool=False):
        return self.get_page_text(self.page_counter - 1, remove_headers)

    def get_page_text(self, page_num: int, remove_headers: bool=False) -> str:
        if remove_headers:
            return '\n'.join(self.data[(self.data['page'] == page_num) & (self.data['left'] > self.boundaries['left']) & (self.data['top'] > self.boundaries['top']) & (self.data['right'] < self.boundaries['right']) & (self.data['bottom'] < self.boundaries['bottom'])].dropna().sort_values(['line', 'left']).groupby(['group', 'col_position', 'line'])['text'].apply(' '.join).groupby(['group', 'col_position']).apply('\n'.join).groupby('group').apply('\n'.join))
        else:
            return '\n'.join(self.data[self.data['page'] == page_num].sort_values(['line', 'left']).dropna().groupby(['group', 'col_position', 'line'])['text'].apply(' '.join).groupby(['group', 'col_position']).apply('\n'.join).groupby('group').apply('\n'.join))

    def get_data(self):
        return self.data

    def get_page_data(self, page_num: int):
        return self.data[self.data['page'] == page_num]

    def save_data(self, filename: str):
        self.data.to_csv(filename, index=False)

    def save_page_data(self, page_num: int, filename: str):
        self.data[self.data['page'] == page_num].to_csv(filename, index=False)

    def load_data(self, filename: str):
        self.data = pd.read_csv(filename, sep=',')

    def is_empty(self):
        if self.page_counter == 0:
            return True
        else:
            return False
