# Simple RAG for Norm-like files

This repository implements basic features of RAG for normativity-like files:

* PDF document loading:
    * From text (using pypdf).
    * Using Optical Character Recognition (using pytesseract).
    * Combining plain text and OCR.
* PDF document splitting:
    * Creating a tree of sections
* Embeddings creation:
    * Using SentenceTransformers
* Embeddings storage:
    * Storage using ChromaDB

# Scripts

This repository provides two scripts to use use the functionalities through a CLI.

## ExtractorCLI.py

This script loads a PDF file (or multiple files in a directory) and combines plain text and
OCR detection to generate a plain text output or a csv file with information about the
layout of the file. This layout information can be latter passed to the EmbeddingsCLI.py
script to extract the embeddings.

### Examples

Process a PDF file and show its text in console:

    python ExtractorCLI.py -f /path/to/file.pdf

Process a PDF file and save the text in a file:

    python ExtractorCLI.py -f /path/to/file.pdf -o /path/to/out.txt

Process a single page of a PDF file:

    python ExtractorCLI.py -f /path/to/file.pdf -p <page>

Process a PDF file and save the data as CSV for latter processing:

    python ExtractorCLI.py -f /path/to/file.pdf -t csv -o /path/to/out.csv

Process all the PDF files in a directory and save a .txt file for each .pdf file:

    python ExtractorCLI.py -d /path/to/dir/ -o /path/to/out/

Process all the PDF files in a directory and save a .csv file for each .pdf file:

    python ExtractorCLI.py -d /path/to/dir/ -t csv -o /path/to/out/

## EmbeddingsCLI.py

This script load CSV data obtained with ExtractorCLI.py and splits the file in sections
creating a tree structure of the document. It can also calculate the embeddings for
each node of the tree or each paragraph of the document and save it as csv or save it
in a vector database creating a collection.

### Examples

Load CSV data and show the tree of the document as an image:

    python EmbeddingsCLI.py -f /path/to/file.csv -a tree

Load CSV data and show the tree structure of the document as a string:

    python EmbeddingsCLI.py -f /path/to/file.csv -a structure

Load CSV data and save the tree of the document as an image:

    python EmbeddingsCLI.py -f /path/to/file.csv -a tree -o /path/to/out.png

Load CSV data, calculate the embeddings of each section and save it in a collection
using ChromaDB:

    python EmbeddingsCLI.py -f /path/to/file.csv -a embeddings -e AllMiniLM -c <CollectionName> -s chromadb --inner-splitter section

Load CSV data, calculate the embeddings of each paragraph and save it in a collection
using ChromaDB:

    python EmbeddingsCLI.py -f /path/to/file.csv -a embeddings -e AllMiniLM -c <CollectionName> -s chromadb --inner-splitter paragraph

# Library

In case you want to use the modules directly from code here is a full example usage
to load PDF file and combine plain text and OCR information to get the embeddings:

    from document_loaders.pdf import PyPDFMixedLoader
    from embeddings.embedders import AllMiniLM
    from embeddings.storage import ChromaDBStorage

    pdf_loader = PyPDFMixedLoader()

    # Load file and merge all information
    pdf_loader.load("/path/to/file.pdf")

    # Get document data as PdfDocumentData
    data = pdf_loader.get_document_data()

    # Create a tree with the structure of the document
    splitter = TreeSplitter(data.get_data(remove_headers=True), filename)
    splitter.analyze()

    # Split document in sentences
    sentences = []
    metadatas = []
    documents = splitter.extract_documents(self._args.inner_splitter)
    for doc in documents:
        sentences.append(doc.get_content())
        metadatas.append(doc.get_metadata())

    # (Optional) In case you need the embeddings you can calculate them
    embedder = AllMiniLM()
    embeddings = embed.get_embeddings(sentences)

    # Store embeddings
    storage = ChromaDBStorage()
    storage.save_info("CollectionName", sentences, metadatas, embeddings)

