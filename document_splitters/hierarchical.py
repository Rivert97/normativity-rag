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

    def append_name(self, name:set):
        """Append content to the name of the document."""
        self.name += ' ' + name

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
    prev_type: float = 0

@dataclass
class TreeState:
    """Store state while handling the tree construction of the document."""
    current_nodes: list[DocNode]|None = None
    last_node: DocNode|None = None
    block_words: pd.DataFrame|None = None

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

    def extract_documents(self, inner_splitter: str) -> list[dict[str:str|dict]]:
        """Get a list of documents corresponding to each node of the tree and its substrings."""
        documents = []
        for node in PreOrderIter(self.root):
            splits = node.split_content(remove_hypens=True, split_type=inner_splitter)
            for split in splits:
                doc = {
                    'content': split,
                    'metadata': {
                        'document_name': self.document_name,
                        'title': node.get_full_title(),
                        'path': node.get_path(),
                    }
                }
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

        self.data = data.copy().dropna()

        self.writable_width = self.data['right'].max() - self.data['left'].min()
        self.line_height = (self.data['bottom'] - self.data['top']).mean()
        self.detector = TitleDetector()

    def analyze(self):
        """Analyze the data to generate the tree."""
        self.__assign_block()
        self.__create_tree_structure()

    def __assign_block(self):
        self.data['block'] = pd.Series(dtype=int)
        prev_line_height = (self.data['top'] - self.data['bottom']).mean()
        block = -1
        for _, page_words in self.data.groupby('page'):
            prev_y = 0
            for _, group_words in page_words.groupby('group'):
                for _, col_words in group_words.groupby('col_position'):
                    lines = col_words.groupby('line')
                    for n_line, line_words in lines:
                        top = line_words['top'].min()
                        bottom = line_words['bottom'].max()
                        line_height = bottom - top

                        if abs(top - prev_y) > min(line_height, prev_line_height) * 1.1:
                            block += 1

                        self.data.loc[lines.groups[n_line], 'block'] = block
                        prev_y = bottom
                        prev_line_height = line_height

    def __create_tree_structure(self):
        n_nodes = self.detector.get_number_of_titles()
        n_nodes += self.detector.get_number_of_content_titles()

        state = TreeState(
            current_nodes=[None for _ in range(n_nodes)],
            last_node=self.root,
        )
        state.current_nodes[0] = state.last_node
        mean_line_height=(self.data['bottom']-self.data['top']).mean()

        sorted_lines = self.data.sort_values(['page', 'line', 'col_position', 'left'])
        for n_page, page_words in sorted_lines.groupby('page'):
            page_center = self.__calculate_center(page_words, 'left', 'right')
            if n_page == 0:
                page_writable_width = self.writable_width
            else:
                page_writable_width = page_words['right'].max() - page_words['left'].min()

            for _, block_words in page_words.groupby('block'):
                state.block_words = block_words
                left = block_words['left'].min()
                right = block_words['right'].max()
                line_height = (block_words['bottom']-block_words['top']).mean()
                is_centered = self.__element_is_centered((left, right), page_center,
                                                         page_writable_width, tolerance_rate=0.1)
                lines_are_larger = line_height > mean_line_height*1.1

                if is_centered or lines_are_larger:
                    self.__handle_title_block(state)
                else:
                    self.__handle_non_title_block(state)

    def __calculate_center(self, df:pd.DataFrame, left_col:float, right_col:float) -> float:
        left = df[left_col].min()
        right = df[right_col].max()
        return left + (right - left) * 0.5

    def __element_is_centered(self, x_limits:tuple[float,float], reference_center:float,
                              reference_width:float, tolerance_rate:float=0.2) -> bool:
        min_x = x_limits[0]
        max_x = x_limits[1]
        center_rate = (reference_center - min_x) / (max_x - reference_center)
        column_percentage = (max_x - min_x) / reference_width

        return (min_x < reference_center < max_x and
                abs(1.0 - center_rate) < tolerance_rate and
                column_percentage < 0.95)

    def __handle_title_block(self, state:TreeState):
        block_text = ' '.join(state.block_words['text'])
        title_level = self.detector.get_title_level(block_text)
        parent = self.find_nearest_parent(state.current_nodes, title_level)
        state.last_node = DocNode(block_text, parent=parent)
        state.current_nodes[title_level] = state.last_node
        self.clear_lower_children(state.current_nodes, title_level)

    def __handle_non_title_block(self, state:TreeState):
        subtitle, block_content = self.__find_block_subtitle(state.block_words)
        if subtitle:
            block_text = block_content

            title_level = self.detector.get_title_level(subtitle)
            if title_level > 1:
                parent = self.find_nearest_parent(state.current_nodes, title_level)
                state.last_node = DocNode(subtitle, parent=parent)
                state.current_nodes[title_level] = state.last_node
                self.clear_lower_children(state.current_nodes, title_level)
        else:
            block_text = ' '.join(state.block_words['text'])

        level, name, content = self.detector.detect_content_header(block_text.lower())
        if level == -1:
            state.last_node.append_content(subtitle)
            state.last_node.append_content(block_text)
        else:
            parent = self.find_nearest_parent(state.current_nodes, level)
            state.last_node = DocNode(name, parent=parent)
            state.last_node.append_content(content)
            if subtitle:
                state.last_node.set_title(subtitle)
            state.current_nodes[level] = state.last_node
            self.clear_lower_children(state.current_nodes, level)

    def __find_block_subtitle(self, block_words):
        subtitle = []
        content = []
        left = block_words['left'].min()
        right = block_words['right'].max()
        width = right - left
        center = left + width * 0.5

        has_reached_content = False
        for _, line_words in block_words.groupby('line'):
            line_str = ' '.join(line_words['text'])
            if has_reached_content:
                content.append(line_str)
                continue

            line_left = line_words['left'].min()
            line_right = line_words['right'].max()
            is_centered = self.__element_is_centered((line_left, line_right),
                                                     center,
                                                     width,
                                                     tolerance_rate=0.1)
            if is_centered:
                subtitle.append(line_str)
                continue

            content.append(line_str)
            has_reached_content = True

        return ' '.join(subtitle), ' '.join(content)


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
        n_nodes = self.detector.get_number_of_titles()
        n_nodes += self.detector.get_number_of_content_titles()

        current_nodes = [None for _ in range(n_nodes)]
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
