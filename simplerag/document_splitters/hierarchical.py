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

    def __init__(self, name:str, parent=None, show_title_before=True,
                 name_sep:str='\n'):
        super().__init__()
        self.name = name
        self.parent = parent
        self.show_title_before = show_title_before
        self.name_sep = name_sep

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

    def get_content(self):
        """Get the string of text of the document"""
        return re.sub(r' ?- ?\n', '', self.content)

    def split_content(self, split_type:str='paragraph', max_characters:int=7500):
        """Split the string of text of the document in multiple strings."""
        content = self.get_content()
        content = re.sub(r'([^.:;yo])(\n)', r'\1 ', content)
        content = re.sub(r'((?<!; )y|(?<!; )o)(\n)', r'\1 ', content)

        if split_type == 'paragraph':
            splits = filter(lambda l: l != '', content.split('\n'))
        elif split_type == 'section':
            if len(content) > max_characters:
                splits = self.__split_on_nearest_dot(content, max_characters)
            else:
                splits = [content]
        else:
            splits = [content]

        return splits

    def get_path(self):
        """Get the full path of the node in the tree."""
        return f'{self.separator.join([""] + [str(node.name) for node in self.path])}'

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

    def __split_on_nearest_dot(self, sentence:str, max_characters:int):
        splited_sentence = sentence.split('.\n')
        sentences = ['']
        current_length = 0
        for s in splited_sentence:
            current_length += len(s) + 2
            if current_length < max_characters:
                sentences[-1] += "\n" + s + "."
            else:
                sentences.append(s + ".")
                current_length = len(s) + 1

        return sentences

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

@dataclass
class DataSplitterOptions:
    """Options for DataTreeSplitter class."""

    loader: str = 'any'
    titles_regex: dict[str|int,str] = None
    absolute_center: bool = False
    max_characters: int = 8000

class TreeSplitter():
    """Base class for splitters that genrate a tree of a document."""

    def __init__(self, document_name: str, max_characters: int):
        self.document_name = document_name
        self.max_characters = max_characters

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
            if node.get_content() == '':
                continue

            splits = node.split_content(inner_splitter, self.max_characters)
            for idx, split in enumerate(splits):
                doc = {
                    'content': split,
                    'metadata': {
                        'document_name': self.document_name,
                        'title': node.get_full_title(),
                        'path': node.get_path(),
                        'parent': node.get_path().split('/')[-1],
                        'num': idx,
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

    def get_clean_text(self):
        """Return the reconstructed text from the tree."""
        blocks = []
        for node in PreOrderIter(self.root):
            if node.name == 'root':
                text = ''
            else:
                text = node.name + node.name_sep

            if node.title != '':
                if node.show_title_before:
                    text = node.title + "\n" + text
                else:
                    text += node.title + "\n"
            if node.content != '':
                text += node.get_content() + "\n"

            blocks.append(text)

        return '\n'.join(blocks)


class DataTreeSplitter(TreeSplitter):
    """Class to split a document in sections to generate a tree structure.

    Uses an array of data obtained with OCR to identify titles and centered texts.
    """

    def __init__(self, data: pd.DataFrame, document_name: str = '',
                 options:DataSplitterOptions=None):
        if options is None:
            options = DataSplitterOptions()
        super().__init__(document_name, options.max_characters)

        self.data = data.copy().dropna()
        self.loader = options.loader
        self.absolute_center = options.absolute_center

        self.writable_width = self.data['right'].max() - self.data['left'].min()
        self.detector = TitleDetector(options.titles_regex)
        self.block_tolerance_rate = 1.6 if self.loader == 'mixed' else 0.8

    def analyze(self):
        """Analyze the data to generate the tree."""
        self.__assign_block()
        self.__create_tree_structure()

    def __assign_block(self):
        self.data['block'] = pd.Series(dtype=int)
        prev_line_height = self.data.groupby('line')[['bottom', 'top']].apply(
            lambda g: g['bottom'].max() - g['top'].min()
        ).mean()
        block = -1
        for _, page_words in self.data.groupby('page'):
            prev_y = 0
            for _, group_words in page_words.groupby('group'):
                for _, col_words in group_words.groupby('col_position'):
                    lines = col_words.groupby('line')
                    for n_line, line_words in lines:
                        top = line_words['top'].min()
                        bottom = line_words['bottom'].mode()[0]
                        line_height = bottom - top

                        if (abs(top - prev_y) >
                            min(line_height, prev_line_height) * self.block_tolerance_rate):
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
        mean_line_height=self.data.groupby('line')[['bottom', 'top']].apply(
            lambda g: g['bottom'].max() - g['top'].min()
        ).mean()

        sorted_lines = self.data.sort_values(['page', 'line', 'col_position', 'left'])
        for _, page_words in sorted_lines.groupby('page'):
            if self.absolute_center:
                reference_x = (0.0, 1.0)
                max_column_percentage = 0.95 - (1.0 - page_words['right'].max()) * 2.0
            elif len(page_words.groupby('line')) < 2:
                reference_x = (self.data['left'].min(), self.data['right'].max())
                max_column_percentage = 0.95
            else:
                reference_x = (page_words['left'].min(), page_words['right'].max())
                max_column_percentage = 0.95

            for _, block_words in page_words.groupby('block'):
                state.block_words = block_words
                left = block_words['left'].min()
                right = block_words['right'].max()
                line_height = block_words.groupby('line')[['bottom', 'top']].apply(
                    lambda g: g['bottom'].max() - g['top'].min()
                ).min()
                is_centered = self.__element_is_centered((left, right), reference_x,
                                                         0.1, max_column_percentage)
                lines_are_larger = line_height > mean_line_height*1.15

                if is_centered or lines_are_larger:
                    self.__handle_title_block(state)
                else:
                    self.__handle_non_title_block(state)

    def __element_is_centered(self, x_limits:tuple[float,float], reference_x:tuple[float,float],
                              tolerance_rate:float=0.2, max_column_percentage:float=0.95) -> bool:
        min_x = x_limits[0]
        max_x = x_limits[1]
        reference_width = reference_x[1] - reference_x[0]
        reference_center = reference_x[0] + reference_width * 0.5
        center_rate = (reference_center - min_x) / (max_x - reference_center)
        column_percentage = (max_x - min_x) / reference_width
        offset_left = min_x - (reference_center - reference_width * 0.5)
        offset_right = (reference_center + reference_width * 0.5) - max_x

        return (min_x < reference_center < max_x and
                abs(1.0 - center_rate) < tolerance_rate and
                column_percentage < max_column_percentage and
                offset_left > 0.01 and
                offset_right > 0.01)

    def __handle_title_block(self, state:TreeState):
        block_text = self.__get_dehypenated_text(state.block_words)
        title_level = self.detector.get_title_level(block_text)
        if title_level == 0:
            self.__handle_non_title_block(state)
        else:
            parent = self.find_nearest_parent(state.current_nodes, title_level)
            state.last_node = DocNode(block_text, parent=parent)
            state.current_nodes[title_level] = state.last_node
            self.clear_lower_children(state.current_nodes, title_level)

    def __handle_non_title_block(self, state:TreeState):
        subtitle, block_text = self.__find_block_subtitle_by_regex(state.block_words)

        if not subtitle:
            subtitle, block_text = self.__find_block_subtitle(state.block_words)
            if subtitle:
                title_level = self.detector.get_title_level(subtitle)
                if title_level > 1:
                    parent = self.find_nearest_parent(state.current_nodes, title_level)
                    state.last_node = DocNode(subtitle, parent=parent, name_sep='. ')
                    state.current_nodes[title_level] = state.last_node
                    self.clear_lower_children(state.current_nodes, title_level)
            else:
                block_text = self.__get_dehypenated_text(state.block_words)

        level, name, content = self.detector.detect_content_header(block_text)
        if level == -1:
            if subtitle:
                state.last_node.append_content(subtitle)
            state.last_node.append_content(block_text)
        else:
            parent = self.find_nearest_parent(state.current_nodes, level)
            state.last_node = DocNode(name, parent=parent, name_sep='. ')
            state.last_node.append_content(content)
            if subtitle:
                state.last_node.set_title(subtitle)
            state.current_nodes[level] = state.last_node
            self.clear_lower_children(state.current_nodes, level)

    def __find_block_subtitle_by_regex(self, block_words, max_subtitle_lines=3):
        block_str = '\n'.join(
            block_words.sort_values('left').groupby('line')['text'].apply(' '.join)
        )
        subtitle, content = self.detector.detect_content_header_with_subtitle(block_str,
                                                                              max_subtitle_lines)

        if subtitle:
            subtitle = self.__get_dehypenated_text_with_str(subtitle)

        content = self.__get_dehypenated_text_with_str(content)

        return subtitle, content

    def __find_block_subtitle(self, block_words):
        subtitle_idx = []
        content_idx = []
        left = block_words['left'].min()
        right = block_words['right'].max()

        has_reached_content = False
        for _, line_words in block_words.groupby('line'):
            if has_reached_content:
                content_idx.extend(line_words.index)
                continue

            line_left = line_words['left'].min()
            line_right = line_words['right'].max()
            is_centered = self.__element_is_centered((line_left, line_right),
                                                     (left, right),
                                                     tolerance_rate=0.1)
            if is_centered:
                subtitle_idx.extend(line_words.index)
                continue

            is_right_aligned = self.__element_is_right_aligned((line_left, line_right),
                                                                (left, right),
                                                                max_percentage=0.9)
            if is_right_aligned:
                subtitle_idx.extend(line_words.index)
                continue

            content_idx.extend(line_words.index)
            has_reached_content = True

        if subtitle_idx:
            subtitle = self.__get_dehypenated_text(block_words.loc[subtitle_idx])
        else:
            subtitle = ''
        content = self.__get_dehypenated_text(block_words.loc[content_idx])

        return subtitle, content

    def __get_dehypenated_text(self, block_words:pd.DataFrame):
        text_lines = '\n'.join(block_words.groupby('line')['text'].apply(' '.join))
        dehypenated_text = re.sub(r' ?- ?\n', '', text_lines)
        joined_lines_text = re.sub(r'([^.:;yo])(\n)', r'\1 ', dehypenated_text)
        joined_lines_text = re.sub(r'((?<!; )y|(?<!; )o)(\n)', r'\1 ', joined_lines_text)

        return joined_lines_text

    def __get_dehypenated_text_with_str(self, text_lines:str):
        dehypenated_text = re.sub(r' ?- ?\n', '', text_lines)
        joined_lines_text = re.sub(r'([^.:;yo])(\n)', r'\1 ', dehypenated_text)
        joined_lines_text = re.sub(r'((?<!; )y|(?<!; )o)(\n)', r'\1 ', joined_lines_text)

        return joined_lines_text

    def __element_is_right_aligned(self, x_limits:tuple[float,float],
                                   ref_x_limits:tuple[float,float], max_percentage:float=0.6):
        offset_right = ref_x_limits[1] - x_limits[1]
        occupancy_percentage = (x_limits[1] - x_limits[0]) / (ref_x_limits[1] - ref_x_limits[0])
        return offset_right < 0.001 and occupancy_percentage < max_percentage

class TextTreeSplitter(TreeSplitter):
    """Class to split a document in sections to generate a tree structure.

    Uses raw text to identify titles.
    """

    def __init__(self, text: str, document_name: str = '', max_characters: int = 7500):
        super().__init__(document_name, max_characters)

        self.text = text

        self.detector = TitleDetector()

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
                level, name, content = self.detector.detect_content_header(line_text)

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
                new_node = DocNode(line_text, parent=parent, name_sep='. ')
                current_nodes[title_level] = new_node
                self.__clear_lower_children(current_nodes, title_level)
                last_node = new_node


    def __find_nearest_parent(self, nodes:list[DocNode], level:int):
        return next(node for node in nodes[level - 1::-1] if node is not None)

    def __clear_lower_children(self, nodes:list[DocNode], from_level:int):
        for i in range(from_level + 1, len(nodes)):
            nodes[i] = None
