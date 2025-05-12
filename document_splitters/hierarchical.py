import pandas as pd
import re
from anytree import RenderTree, NodeMixin, PreOrderIter
from anytree.node.node import _repr
from anytree.exporter import UniqueDotExporter

from .data import Document
from .detectors import TitleDetector

class DocNode(NodeMixin):

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

    def append_title(self, title:str):
        if self.title == '':
            self.title = title
        else:
            self.title += ' ' + title

    def get_full_title(self):
        text = self.name

        if self.title != '':
            text += f' ({self.title})'

        return text

    def append_content(self, content:str):
        if self.content == '':
            self.content = content
        else:
            self.content += "\n" + content

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

class TreeSplitter:

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

        self.writable_width = self.data['right'].max() - self.data['left'].min()
        self.line_height = (self.data['bottom'] - self.data['top']).mean()
        self.detector = TitleDetector()

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

    def get_file_structure(self):
        text = ''
        for pre, fill, node in RenderTree(self.root):
            text += (f'{pre} {node}\n')

        return text

    def show_file_structure(self):
        print(self.get_file_structure())

    def show_tree(self, filename:str):
        UniqueDotExporter(self.root).to_picture(filename)

    def __assign_title_type(self):
        self.data['title_type'] = pd.Series(dtype=int)
        for page, page_words in self.data.groupby('page'):
            page_values = page_words.agg({'left': ['min'], 'right': ['max']})
            page_center = page_values['left']['min'] + (page_values['right']['max'] - page_values['left']['min']) * 0.5

            for group, group_words in page_words.groupby('group'):
                num_cols = group_words['column'].max() + 1
                for column, column_words in group_words.groupby('col_position'):
                    values_col = column_words.agg({'left': ['min'], 'right': ['max']})
                    column_width = values_col['right']['max'] - values_col['left']['min']
                    column_center = values_col['left']['min'] + column_width * 0.5
                    prev_y = 0
                    prev_level = 0
                    for line, line_words in column_words.groupby('line'):
                        values = line_words.agg({'left': ['min'], 'right': ['max'], 'top': ['min'], 'bottom': ['max']})
                        if num_cols == 1:
                            if self.__element_is_centered(values['left']['min'], values['right']['max'], page_center, self.writable_width):
                                title_level = 1
                            else:
                                title_level = 0
                        else:
                            if (self.__element_is_centered(values['left']['min'], values['right']['max'], column_center, column_width)
                                and (self.__element_is_vertically_separated(values['top']['min'], prev_y) or prev_level == 2)):
                                title_level = 2
                            else:
                                title_level = 0
                        self.data.loc[line_words.index, 'title_type'] = title_level
                        prev_y = values['bottom']['max']
                        prev_level = title_level

    def __assign_title_level(self):
        self.data['title_level'] = pd.Series(dtype=int)
        self.data['title_level'] = 0
        for line, line_words in self.data[self.data['title_type'] == 1].sort_values('left').groupby('line'):
            text = ' '.join(line_words['text'])
            self.data.loc[line_words.index, 'title_level'] = self.detector.get_title_level(text)

    def __element_is_centered(self, min_x:float, max_x:float, reference_center:float, reference_width:float):
        center_rate = (reference_center - min_x) / (max_x - reference_center)
        column_percentage = (max_x - min_x) / reference_width
        if min_x < reference_center < max_x and abs(1.0 - center_rate) < 0.2 and column_percentage < 0.8:
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
        n_titles = self.detector.get_number_of_titles()
        n_content_titles = self.detector.get_number_of_content_tiltes()

        self.root = DocNode("root")
        current_nodes = [None for _ in range(n_titles + n_content_titles)]
        last_node = self.root
        current_nodes[0] = last_node

        prev_was_title = False
        prev_y = 0
        future_title = ''
        for line, line_words in self.data.sort_values(['page', 'line', 'left']).groupby(['page', 'group', 'col_position', 'line']):
            line_str = ' '.join(line_words['text'])
            title_type = line_words.iloc[0]['title_type']

            if title_type == 1:
                if self.__element_is_vertically_separated(line_words['top'].min(), prev_y):
                    title_level = self.detector.get_title_level(line_str)
                    parent = next(node for node in current_nodes[title_level - 1::-1] if node is not None)
                    last_node = DocNode(line_str, parent=parent)
                    current_nodes[title_level] = last_node
                    current_nodes[title_level+1:] = [None for _ in range(len(current_nodes[title_level+1:]))]
                    prev_was_title = True
                elif prev_was_title:
                    last_node.append_title(line_str)
            elif title_type == 2:
                future_title += line_str + " "
                prev_was_title = False
            else:
                level, name, content = self.detector.detect_content_header(line_str.lower())
                if level == -1: # No content header found
                    last_node.append_content(content)
                else:
                    parent = next(node for node in current_nodes[level - 1::-1] if node is not None)
                    last_node = DocNode(name, parent=parent)
                    last_node.append_content(content)
                    if future_title != '':
                        last_node.set_title(future_title.strip())
                        future_title = ''
                    current_nodes[level] = last_node
                    current_nodes[level+1:] = [None for _ in range(len(current_nodes[level+1:]))]

                prev_was_title = False

            prev_y = line_words['bottom'].max()