from typing import List, Tuple
import pandas as pd
import re
import os
from PIL import Image

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

    def __init__(self, data: pd.DataFrame, document_name: str = ''):
        self.data = data.copy()
        self.document_name = document_name

        self.writable_width = self.data['right'].max() - self.data['left'].min()
        self.line_height = (self.data['bottom'] - self.data['top']).mean()
        self.detector = TitleDetector()

        self.root = None

    def analyze(self):
        self.__assign_title_type()
        self.__create_tree_structure()

    def extract_documents(self):
        documents = []
        for node in PreOrderIter(self.root):
            splits = node.split_content(remove_hypens=True)
            for split in splits:
                doc = Document(
                    content=split,
                    metadata={
                        'document_name': self.document_name,
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

    def show_tree(self):
        filename = '.tree.png'
        UniqueDotExporter(self.root).to_picture(filename)
        img = Image.open(filename)
        img.show()

        os.remove(filename)

    def save_tree(self, filename:str):
        UniqueDotExporter(self.root).to_picture(filename)

    def __assign_title_type(self):
        self.data['title_type'] = pd.Series(dtype=int)

        for page, page_words in self.data.groupby('page'):
            page_center = self.__calculate_center(page_words, 'left', 'right')

            for group, group_words in page_words.groupby('group'):
                num_cols = group_words['column'].max() + 1

                for col_pos, col_words in group_words.groupby('col_position'):
                    col_center = self.__calculate_center(col_words, 'left', 'right')
                    col_width = col_words['right'].max() - col_words['left'].min()

                    prev_y = 0
                    prev_level = 0

                    for line, line_words in col_words.groupby('line'):
                        title_level, prev_y, prev_level = self.__process_line(
                            line_words,
                            num_cols,
                            page_center,
                            col_center,
                            col_width,
                            prev_y,
                            prev_level
                        )
                        self.data.loc[line_words.index, 'title_type'] = title_level

    def __calculate_center(self, df:pd.DataFrame, left_col:int, right_col:int):
        left = df[left_col].min()
        right = df[right_col].max()
        return left + (right - left) * 0.5

    def __process_line(self, line_words, num_columns, page_center, col_center, col_width, prev_y, prev_level):
        left = line_words['left'].min()
        right = line_words['right'].max()
        top = line_words['top'].min()
        bottom = line_words['bottom'].max()

        if num_columns == 1:
            is_centered = self.__element_is_centered(left, right, page_center, self.writable_width)
            if is_centered:
                title_level = 1
            else:
                title_level = 0
        else:
            is_centered = self.__element_is_centered(left, right, col_center, col_width)
            is_separated = self.__element_is_vertically_separated(top, prev_y)
            if is_centered and (is_separated or prev_level == 2):
                title_level = 2
            else:
                title_level = 0

        return title_level, bottom, title_level

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

        sorted_lines = self.data.sort_values(['page', 'line', 'left'])
        for line, line_words in sorted_lines.groupby(['page', 'group', 'col_position', 'line']):
            line_text = ' '.join(line_words['text'])
            title_type = line_words.iloc[0]['title_type']
            top = line_words['top'].min()
            bottom = line_words['bottom'].max()

            if title_type == 1:
                last_node, prev_was_title = self.__handle_title_type_1(
                    line_text, top, prev_y, current_nodes, last_node, prev_was_title
                )
            elif title_type == 2:
                future_title += line_text + " "
                prev_was_title = False
            else:
                last_node, future_title = self.__handle_content_line(
                    line_text, current_nodes, last_node, future_title
                )
                prev_was_title = False

            prev_y = bottom

    def __handle_title_type_1(self, line_text:str, current_top:float, prev_y:float, current_nodes:List[DocNode], last_node:DocNode, prev_was_title:bool) -> Tuple[DocNode, bool]:
        if self.__element_is_vertically_separated(current_top, prev_y):
            title_level = self.detector.get_title_level(line_text)
            parent = self.__find_nearest_parent(current_nodes, title_level)
            new_node = DocNode(line_text, parent=parent)
            current_nodes[title_level] = new_node
            self.__clear_lower_children(current_nodes, title_level)
            return new_node, True
        elif prev_was_title:
            last_node.append_title(line_text)

        return last_node, prev_was_title

    def __handle_content_line(self, line_text:str, current_nodes:List[DocNode], last_node:DocNode, future_title:str):
        level, name, content = self.detector.detect_content_header(line_text.lower())
        if level == -1:
            last_node.append_content(content)
        else:
            parent = self.__find_nearest_parent(current_nodes, level)
            new_node = DocNode(name, parent=parent)
            new_node.append_content(content)
            if future_title.strip():
                new_node.set_title(future_title.strip())
                future_title = ''
            current_nodes[level] = new_node
            self.__clear_lower_children(current_nodes, level)
            last_node = new_node

        return last_node, future_title

    def __find_nearest_parent(self, nodes:List[DocNode], level:int):
        return next(node for node in nodes[level - 1::-1] if node is not None)

    def __clear_lower_children(self, nodes:List[DocNode], from_level:int):
        for i in range(from_level + 1, len(nodes)):
            nodes[i] = None