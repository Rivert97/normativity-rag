"""Module to load PDF files and extract it's text or data."""
from dataclasses import dataclass
from enum import Enum
import difflib
import re
import math

import pandas as pd

from .parsers import PypdfParser, OcrPdfParser, PypdfPage, OcrPage, PdfPlumberParser
from .representations import PdfDocumentData
from .processors import remove_hyphens, replace_ligatures

class Status(Enum):
    """Different types of status that can be present while mergin differences."""
    NA = -1
    EQUAL = 0
    ADDITION = 1
    REMOVAL = 2
    ANNOTATION_ADD_OR_REM = 3
    ANNOTATION_CHANGE_START = 4
    ANNOTATION_CHANGE_END = 5

@dataclass
class LoaderOptions:
    """Options to be passed to the different Loaders."""

    cache_dir: str = './.cache'
    keep_cache: bool = False
    visual_aid: bool = False

@dataclass
class DifferenceState:
    """Store state when making matches between differences from the text."""

    merged: list
    added_words: list
    removed_words: list
    missing_additions: list[list[(int, str)]]
    missing_removals: list[list]
    ocr_idx: int
    status_tracker: list

class PyPDFLoader():
    """Class to load a PDF file using pypdf library.

    This class was created only for compatibility with the extraction script.
    """
    def __init__(self, file_path:str):
        """Open the document."""
        self.parser = PypdfParser(file_path)

    def get_text(self, remove_headers:bool=True, boundaries:dict[str,float]=None):
        """Return the full text of the PDF file."""
        text = self.parser.get_text(remove_headers=remove_headers, boundaries=boundaries)
        text = replace_ligatures(text)
        text = remove_hyphens(text)

        return text

    def get_page_text(self, page_num: int, remove_headers:bool=True,
                      boundaries:dict[str,float]=None):
        """Return the text of a single page of the PDF file."""
        page = self.parser.get_page(page_num)
        page_text = page.get_text(remove_headers, boundaries)
        page_text = replace_ligatures(page_text)
        page_text = remove_hyphens(page_text)

        return page_text

class OCRLoader():
    """Class to load a PDF file using pytesseract to extract the text through OCR."""
    def __init__(self, file_path:str, options:LoaderOptions=None):
        """Open the document."""
        if options is None:
            options = LoaderOptions()

        self.parser = OcrPdfParser(file_path, options.cache_dir, options.keep_cache,
                                   options.visual_aid)

    def get_text(self, remove_headers:bool=True, boundaries:dict[str,float]=None) -> str:
        """Return the full text of the PDF file."""
        text = self.parser.get_text(remove_headers=remove_headers, boundaries=boundaries)
        text = replace_ligatures(text)
        text = remove_hyphens(text)

        return text

    def get_page_text(self, page_num: int, remove_headers:bool=True,
                      boundaries:dict[str,float]=None) -> str:
        """Return the text of a single page of the PDF file."""
        page = self.parser.get_page(page_num)
        page_text = page.get_text(remove_headers=remove_headers, boundaries=boundaries)
        page_text = replace_ligatures(page_text)
        page_text = remove_hyphens(page_text)

        return page_text

    def get_document_data(self):
        """Return the relevant OCR data of the hole document."""
        document_data = PdfDocumentData()
        for page in self.parser.get_pages():
            document_data.add_page(page.get_data())

        return document_data

class PDFPlumberLoader():
    """Class to load a PDF file using pdfplumber with custom text reconstruction."""
    def __init__(self, file_path:str, raw:bool=False, options:LoaderOptions=None):
        if options is None:
            options = LoaderOptions()

        self.parser = PdfPlumberParser(file_path, options.cache_dir, options.keep_cache,
                                       options.visual_aid)
        self.raw = raw

    def get_text(self, remove_headers:bool=True, boundaries:dict[str,float]=None) -> str:
        """Return the full text of the PDF file."""
        if self.raw:
            text = self.parser.get_raw_text(remove_headers=remove_headers, boundaries=boundaries)
        else:
            text = self.parser.get_text(remove_headers=remove_headers, boundaries=boundaries)
        text = replace_ligatures(text)
        text = remove_hyphens(text)

        return text

    def get_page_text(self, page_num: int, remove_headers:bool=True,
                      boundaries:dict[str,float]=None) -> str:
        """Return the text of a single page of the PDF file."""
        page = self.parser.get_page(page_num)
        if self.raw:
            page_text = page.get_raw_text(remove_headers, boundaries)
        else:
            page_text = page.get_text(remove_headers, boundaries)
        page_text = replace_ligatures(page_text)
        page_text = remove_hyphens(page_text)

        return page_text

    def get_document_data(self):
        """Return the relevant data of the hole document."""
        document_data = PdfDocumentData()
        for page in self.parser.get_pages():
            document_data.add_page(page.get_data())

        return document_data
