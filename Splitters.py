from typing import Dict
import pandas as pd
import re
from anytree import RenderTree, NodeMixin, PreOrderIter
from anytree.node.node import _repr
from anytree.exporter import UniqueDotExporter

class Document:

    def __init__(self, content:str, metadata:Dict={}):
        self.content = content
        self.metadata = metadata

    def get_content(self):
        return self.content

    def get_metadata(self):
        return self.metadata

    def __str__(self):
        return f"Document(content='{self.content}', metadata={{{', '.join([f"{key}:{value}" for key, value in self.metadata.items()])}}})"

class SplitNode(NodeMixin):

    def __init__(self, name:str, parent=None, children=None):
        super().__init__()
        self.name = name
        self.parent = parent
        if children:
            self.children = children

        self.title = ''
        self.content = ''

    def set_title(self, title:str):
        self.title = title

    def get_full_title(self):
        text = self.name

        if self.title != '':
            text += f' ({self.title})'

        return text

    def add_content(self, content:str):
        self.content += content + "\n"

    def get_content(self, remove_hypens:bool=True):
        if remove_hypens:
            return re.sub(r' ?- ?\n', '', self.content)
        else:
            return self.content

    def split_content(self, remove_hypens:bool=True):
        content = self.get_content(remove_hypens)

        content = re.sub(r'([^.:;\n])(\n)', r'\1 ', content)

        return content.split('\n')

    def get_path(self):
        return "{!r}".format(self.separator.join([""] + [str(node.name) for node in self.path]))

    def __str__(self):
        text = self.get_full_title()

        content = self.get_content()
        if content != '':
            if len(content) > 15:
                text += f': {content[:15]}...'
            else:
                text += f': {content}'

        return text

    def __repr__(self):
        args = ["{!r}".format(self.separator.join([""] + [str(node.name) for node in self.path]))]
        return _repr(self, args=args, nameblacklist=["name"])

class NormativitySplitter:

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

        self.writable_width = self.data['right'].max() - self.data['left'].min()
        self.line_height = (self.data['bottom'] - self.data['top']).mean()
        self.metadata = {
            'titles': {
                2: r'^(t[iíÍ]tulo|[xiv]+\.) .*',
                3: r'^cap[iíÍ]tulo .*',
            },
            'contents': {
                0: r'^art[iíÍ]culo ([0-9]+|[a-z]+(ro|do|ro|to|mo|vo|no))(bis|ter|qu[aá]ter|quinquies)?\.',
            }
        }

        self.root = None

    def analyze(self):
        self.__assign_title_type()
        self.__assign_title_level()
        self.__create_tree_structure()

    def extract_documents(self):
        documents = []
        for node in PreOrderIter(self.root):
            splits = node.split_content(remove_hypens=True)
            for split in splits:
                doc = Document(
                    content=split,
                    metadata={
                        'title': node.get_full_title(),
                        'path': node.get_path(),
                    }
                )
                documents.append(doc)

        return documents

    def show_file_structure(self):
        for pre, fill, node in RenderTree(self.root):
            print(f'{pre} {node}')

    def show_tree(self):
        UniqueDotExporter(self.root).to_picture('outs/tree.png')

    def __assign_title_type(self):
        self.data['title_type'] = pd.Series(dtype=int)
        for page, page_words in self.data.groupby('page'):
            page_values = page_words.agg({'left': ['min'], 'right': ['max']})
            page_center = page_values['left']['min'] + (page_values['right']['max'] - page_values['left']['min']) * 0.5

            for group, group_words in page_words.groupby('group'):
                num_cols = group_words['column'].max() + 1
                for column, column_words in group_words.groupby('column'):
                    values_col = column_words.agg({'left': ['min'], 'right': ['max']})
                    column_width = values_col['right']['max'] - values_col['left']['min']
                    column_center = values_col['left']['min'] + column_width * 0.5
                    for line, line_words in column_words.groupby('line'):
                        values = line_words.agg({'left': ['min'], 'right': ['max'], 'top': ['min'], 'bottom': ['max']})
                        if num_cols == 1:
                            if self.__element_is_centered(values['left']['min'], values['right']['max'], page_center, self.writable_width):
                                self.data.loc[line_words.index, 'title_type'] = 1
                            else:
                                self.data.loc[line_words.index, 'title_type'] = 0
                        else:
                            if self.__element_is_centered(values['left']['min'], values['right']['max'], column_center, column_width):
                                self.data.loc[line_words.index, 'title_type'] = 2
                            else:
                                self.data.loc[line_words.index, 'title_type'] = 0

    def __assign_title_level(self):
        self.data['title_level'] = pd.Series(dtype=int)
        self.data['title_level'] = 0
        for line, line_words in self.data[self.data['title_type'] == 1].sort_values('left').groupby('line'):
            text = ' '.join(line_words['text'])
            for lvl in self.metadata['titles']:
                if re.match(self.metadata['titles'][lvl], text.lower()):
                    self.data.loc[line_words.index, 'title_level'] = lvl
                    break
            else:
                self.data.loc[line_words.index, 'title_level'] = 1

    def __element_is_centered(self, min_x:float, max_x:float, reference_center:float, reference_width:float):
        center_rate = (reference_center - min_x) / (max_x - reference_center)
        column_percentage = (max_x - min_x) / reference_width
        if min_x < reference_center < max_x and abs(1.0 - center_rate) < 0.1 and column_percentage < 0.9:
            return True
        else:
            return False

    def __element_is_vertically_separated(self, pos_y:float, prev_pos_y:float):
        tolerance = self.line_height * 1.5

        if abs(pos_y - prev_pos_y) > tolerance:
            return True
        else:
            return False

    def __create_tree_structure(self):
        n_titles = len(self.metadata['titles']) + 2
        n_contents = len(self.metadata['contents'])

        self.root = SplitNode("root")
        current_chunks = [None for i in range(n_titles + n_contents)]
        current_chunks[0] = self.root
        last_chunk = None
        prev_was_title = False
        prev_y = 0
        for line, line_words in self.data.sort_values(['page', 'line', 'left']).groupby(['page', 'group', 'col_position', 'line']):
            title_type = line_words.iloc[0]['title_type']
            title_level = line_words.iloc[0]['title_level']
            line_str = ' '.join(line_words['text'])
            if title_type == 1:
                if self.__element_is_vertically_separated(line_words['top'].min(), prev_y):
                    for lvl in self.metadata['titles']:
                        if title_level == lvl:
                            last_chunk = SplitNode(line_str, parent=current_chunks[lvl - 1])
                            current_chunks[lvl] = last_chunk
                            break
                    else:
                        last_chunk = SplitNode(line_str, parent=current_chunks[0])
                        current_chunks[1] = last_chunk
                    prev_was_title = True
                elif prev_was_title:
                    last_chunk.set_title(line_str)
            else:
                for lvl in self.metadata['contents']:
                    matches = re.search(self.metadata['contents'][lvl], line_str.lower())
                    if matches:
                        l_match = len(matches.group(0))
                        last_chunk = SplitNode(line_str[:l_match].rstrip('.'), parent=current_chunks[n_titles - 1])
                        last_chunk.add_content(line_str[l_match:].strip())
                        current_chunks[n_titles + lvl]
                        break
                else:
                    last_chunk.content += line_str + '\n'
                prev_was_title = False

            prev_y = line_words['bottom'].max()