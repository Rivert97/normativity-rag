"""Module to define multiple splliters for a text or pdf document.

All the splitters generate a tree with the structure of the document.
"""

import re
import os
from dataclasses import dataclass

from PIL import Image
import pandas as pd
from anytree import RenderTree, NodeMixin, PreOrderIter
from anytree.node.node import _repr
from anytree.exporter import UniqueDotExporter

from .data import Document
from .detectors import TitleDetector

class DocNode(NodeMixin):
    """A simple tree node with additional information of the document.

    A document is a part of the text from a file or long string.
    """

    def __init__(self, name:str, parent=None, children=None):
        super().__init__()
        self.name = name
        self.parent = parent
        if children:
            self.children = children

        self.title = ''
        self.content = ''

    def set_title(self, title:str):
        """Set the title of the document."""
        self.title = title

    def append_title(self, title:str):
        """Append content to the title of the document."""
        if self.title == '':
            self.title = title
        else:
            self.title += ' ' + title

    def get_full_title(self):
        """Return a string that contains the name and title of the node."""
        text = self.name

        if self.title != '':
            text += f' ({self.title})'

        return text

    def append_content(self, content:str):
        """Append content to the string of text of the document."""
        if self.content == '':
            self.content = content
        else:
            self.content += "\n" + content

    def prepend_content(self, content:str):
        """Add content to the start the string of text of the document."""
        if self.content == '':
            self.content = content
        else:
            self.content = content + "\n" + self.content

    def get_content(self, remove_hypens:bool=True):
        """Get the string of text of the document"""
        if remove_hypens:
            return re.sub(r' ?- ?\n', '', self.content)

        return self.content

    def split_content(self, remove_hypens:bool=True, split_type:str='paragraph'):
        """Split the string of text of the document in multiple strings."""
        content = self.get_content(remove_hypens)
        content = re.sub(r'([^.:;\n])(\n)', r'\1 ', content)

        if split_type == 'paragraph':
            splits = content.split('\n')
        elif split_type == 'section':
            splits = [content]
        else:
            splits = [content]

        return splits

    def get_path(self):
        """Get the full path of the node in the tree."""
        return f'{(self.separator.join([""] + [str(node.name) for node in self.path]))!r}'

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
        args = [f'{(self.separator.join([""] + [str(node.name) for node in self.path]))}']
        return _repr(self, args=args, nameblacklist=["name"])

@dataclass
class LineState:
    """Store state while checking the types of titles in the document."""
    num_cols: int = 0
    page_center: float = 0
    col_center: float = 0
    col_width: float = 0
    prev_y: float = 0
    prev_level: float = 0

@dataclass
class TitleState:
    """Store state while handling the levels of titles in the document."""
    current_nodes: list[DocNode]|None = None
    last_node: DocNode = None
    current_top: float = 0
    prev_was_title: bool = False
    prev_y: float = 0
    future_title: str = ''

class TreeSplitter():
    """Base class for splitters that genrate a tree of a document."""

    def __init__(self, document_name: str):
        self.document_name = document_name

        self.root = DocNode("root")

    def get_file_structure(self):
        """Get a formatted string of text that shows the structure of the tree."""
        text = ''
        for pre, _, node in RenderTree(self.root):
            text += (f'{pre} {node}\n')

        return text

    def extract_documents(self, inner_splitter: str):
        """Get a list of documents corresponding to each node of the tree and its substrings."""
        documents = []
        for node in PreOrderIter(self.root):
            splits = node.split_content(remove_hypens=True, split_type=inner_splitter)
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

    def show_file_structure(self):
        """Print a formatted string that shows the structure of the document."""
        print(self.get_file_structure())

    def show_tree(self):
        """Show an image of the tree that represents the document."""
        filename = '.tree.png'
        UniqueDotExporter(self.root).to_picture(filename)
        img = Image.open(filename)
        img.show()

        os.remove(filename)

    def save_tree(self, filename:str):
        """Save an image of the tree that represents the document."""
        UniqueDotExporter(self.root).to_picture(filename)

    def find_nearest_parent(self, nodes:list[DocNode], level:int):
        """Utility function to find the closest not null node in a list starting from the
        specified index or level."""
        return next(node for node in nodes[level - 1::-1] if node is not None)

    def clear_lower_children(self, nodes:list[DocNode], from_level:int):
        """Clear all nodes in the list that are below the specified level."""
        for i in range(from_level + 1, len(nodes)):
            nodes[i] = None

class DataTreeSplitter(TreeSplitter):
    """Class to split a document in sections to generate a tree structure.

    Uses an array of data obtained with OCR to identify titles and centered texts.
    """

    def __init__(self, data: pd.DataFrame, document_name: str = ''):
        super().__init__(document_name)

        self.data = data.copy()

        self.writable_width = self.data['right'].max() - self.data['left'].min()
        self.line_height = (self.data['bottom'] - self.data['top']).mean()
        self.detector = TitleDetector()

    def analyze(self):
        """Analyze the data to generate the tree."""
        self.__assign_title_type()
        self.__create_tree_structure()

    def __assign_title_type(self):
        self.data['title_type'] = pd.Series(dtype=int)

        state = LineState()
        for _, page_words in self.data.groupby('page'):
            state.page_center = self.__calculate_center(page_words, 'left', 'right')

            for _, group_words in page_words.groupby('group'):
                state.num_cols = group_words['column'].max() + 1

                for _, col_words in group_words.groupby('col_position'):
                    state.col_center = self.__calculate_center(col_words, 'left', 'right')
                    state.col_width = col_words['right'].max() - col_words['left'].min()

                    state.prev_y = 0
                    state.prev_level = 0

                    for _, line_words in col_words.groupby('line'):
                        title_level, state.prev_y, state.prev_level = self.__process_line(
                            line_words,
                            state
                        )
                        self.data.loc[line_words.index, 'title_type'] = title_level

    def __calculate_center(self, df:pd.DataFrame, left_col:float, right_col:float) -> float:
        left = df[left_col].min()
        right = df[right_col].max()
        return left + (right - left) * 0.5

    def __process_line(self, line_words:pd.DataFrame, state:LineState):
        left = line_words['left'].min()
        right = line_words['right'].max()
        top = line_words['top'].min()
        bottom = line_words['bottom'].max()

        if state.num_cols == 1:
            is_centered = self.__element_is_centered(left, right, state.page_center,
                                                     self.writable_width)
            if is_centered:
                title_level = 1
            else:
                title_level = 0
        else:
            is_centered = self.__element_is_centered(left, right, state.col_center, state.col_width)
            is_separated = self.__element_is_vertically_separated(top, state.prev_y)
            if is_centered and (is_separated or state.prev_level == 2):
                title_level = 2
            else:
                title_level = 0

        return title_level, bottom, title_level

    def __element_is_centered(self, min_x:float, max_x:float, reference_center:float,
                              reference_width:float):
        center_rate = (reference_center - min_x) / (max_x - reference_center)
        column_percentage = (max_x - min_x) / reference_width

        return (min_x < reference_center < max_x and
                abs(1.0 - center_rate) < 0.2 and
                column_percentage < 0.8)

    def __element_is_vertically_separated(self, pos_y:float, prev_pos_y:float):
        tolerance = self.line_height * 1.5

        return abs(pos_y - prev_pos_y) > tolerance

    def __create_tree_structure(self):
        n_titles = self.detector.get_number_of_titles()
        n_content_titles = self.detector.get_number_of_content_tiltes()

        #current_nodes = [None for _ in range(n_titles + n_content_titles)]
        #last_node = self.root
        #current_nodes[0] = last_node

        #prev_was_title = False
        #prev_y = 0
        #future_title = ''
        state = TitleState(
            current_nodes=[None for _ in range(n_titles + n_content_titles)],
            last_node=self.root,
            current_top=0,
            prev_was_title=False,
            prev_y=0,
            future_title=''
        )
        state.current_nodes[0] = state.last_node

        sorted_lines = self.data.sort_values(['page', 'line', 'left'])
        for _, line_words in sorted_lines.groupby(['page', 'group', 'col_position', 'line']):
            line_text = ' '.join(line_words['text'])
            title_type = line_words.iloc[0]['title_type']
            state.current_top = line_words['top'].min()
            bottom = line_words['bottom'].max()

            if title_type == 1:
                self.__handle_title_type_1(line_text, state)
            elif title_type == 2:
                state.future_title += line_text + " "
                state.prev_was_title = False
            else:
                self.__handle_content_line(line_text, state)
                state.prev_was_title = False

            state.prev_y = bottom

    def __handle_title_type_1(self, line_text:str, state:TitleState):
        if self.__element_is_vertically_separated(state.current_top, state.prev_y):
            title_level = self.detector.get_title_level(line_text)
            parent = self.find_nearest_parent(state.current_nodes, title_level)
            state.last_node = DocNode(line_text, parent=parent)
            state.current_nodes[title_level] = state.last_node
            self.clear_lower_children(state.current_nodes, title_level)
            state.prev_was_title = True

        if state.prev_was_title:
            state.last_node.append_title(line_text)

    def __handle_content_line(self, line_text:str, state:TitleState):
        level, name, content = self.detector.detect_content_header(line_text.lower())
        if level == -1:
            if state.future_title != '':
                state.last_node.prepend_content(state.future_title)
                state.future_title = ''
            state.last_node.append_content(content)
        else:
            parent = self.find_nearest_parent(state.current_nodes, level)
            state.last_node = DocNode(name, parent=parent)
            state.last_node.append_content(content)
            if state.future_title.strip():
                state.last_node.set_title(state.future_title.strip())
                state.future_title = ''
            state.current_nodes[level] = state.last_node
            self.clear_lower_children(state.current_nodes, level)


class TextTreeSplitter(TreeSplitter):
    """Class to split a document in sections to generate a tree structure.

    Uses raw text to identify titles.
    """

    def __init__(self, text: str, document_name: str = ''):
        super().__init__(document_name)

        self.text = text

        self.detector = TitleDetector('not_permissive_titles')

    def analyze(self):
        """Analyze the raw text to generate the tree."""
        n_titles = self.detector.get_number_of_titles()
        n_content_titles = self.detector.get_number_of_content_tiltes()

        current_nodes = [None for _ in range(n_titles + n_content_titles)]
        last_node = self.root
        current_nodes[0] = last_node

        for line_text in self.text.split('\n'):
            title_level = self.detector.get_title_level(line_text)
            if title_level == 1:
                level, name, content = self.detector.detect_content_header(line_text.lower())

                if level == -1:
                    last_node.append_content(content)
                else:
                    parent = self.__find_nearest_parent(current_nodes, level)
                    new_node = DocNode(name, parent=parent)
                    new_node.append_content(content)
                    current_nodes[level] = new_node
                    self.__clear_lower_children(current_nodes, level)
                    last_node = new_node
            else:
                parent = self.__find_nearest_parent(current_nodes, title_level)
                new_node = DocNode(line_text, parent=parent)
                current_nodes[title_level] = new_node
                self.__clear_lower_children(current_nodes, title_level)
                last_node = new_node


    def __find_nearest_parent(self, nodes:list[DocNode], level:int):
        return next(node for node in nodes[level - 1::-1] if node is not None)

    def __clear_lower_children(self, nodes:list[DocNode], from_level:int):
        for i in range(from_level + 1, len(nodes)):
            nodes[i] = None
