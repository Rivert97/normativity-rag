# Simple RAG for Norm-like files

This repository implements basic features of RAG for normativity-like files:

* PDF document loading:
    * From text (using pypdf).
    * Using Optical Character Recognition (using pytesseract).
    * Combining plain text and OCR.
* PDF document splitting:
    * Creating a tree of sections: From plain text or from OCR information.
* Embeddings creation:
    * Using SentenceTransformers
* Embeddings storage:
    * Storage using ChromaDB
*

# Installation

Please refer to the proper installation guide according to your system

* Linux: [Linux Installation Guide](./docs/install/linux.md)
* Windows: Currently not supported
* MaxOS: N/A

# Usage

## run_extractor.py

This script concentrates the processes from *extract_info.py* and *get_embeddings.py*, recieves a PDF file or directory of files as an input and creates a collection of embeddings in a database. This script is intended to be used as the extractor of information without intermediate steps.

This script doesn't create intermediate CSV or txt files, only the final database.

The script can be used passing all the options as paramters to the script or with a configuration file in YAML format.

### Examples

Process a PDF file and save the embeddings in a collection. Use *-k* to keep cache when processing the same file multiple times:

    python run_extractor.py -c <CollectionName> -e "all-MiniLM-L6-v2" --extraction-type data -f /path/to/file.pdf --inner-splitter section -k

Process a directory and save the embeddings in a collection from a settings file:

    python run_extractor.py --settings-file /path/to/settings.yml

Show all the options of the script:

    python run_extractor.py -h

### YALM config example

    db:
        directory: /path/to/dir/

        settings:
            cache_dir: ./.cache
            database_dir: ./db
            keep_cache: True

        collections:
            CollectionName:
                embedder: all-MiniLM-L6-v2
                extraction_type: data
                inner_splitter: section
                loader: mixed

## chat.py

This script allows to chat with a model that performs RAG from a previously created database with *run_extractor.py* or the test scripts.

The script can be used in interactive mode to answer multiple questions in sequence or in single-query mode.

The process followed by the script is as follows:

1. Load the collection of embeddings.
2. Convert the query to embedding and search for similar embeddings in the database to obtain related documents.
3. Load an instruction-following model from HuggingFace, currently available: Gemma, Llama, Qwen, Mistral.
3. Pass the retrieved documents and its metadata as context to the model, alongside the query.
4. Let the model generate the answer and show it to the user.

### Examples

Single-query mode, get the answer to a simple question:

    python chat.py -c CollectionName -d ./db -e "all-MiniLM-L6-v2" -m qwen --variant "3-0.6B" --query "Que tipos de profesores hay?"

NOTE: To know the variant for the model visit HuggingFace models collection (https://huggingface.co/models). A variant is the identifier of the model starting from the version. For model 'google/gemma-3-4b-it' the model is 'GEMMA' and the variant is '3-4b-it'.

Interactive mode:

    python chat.py -c CollectionName -d ./db -e "all-MiniLM-L6-v2" -m GEMMA --variant "3-4b-it"

Interactive mode with context. This option shows the relevant documents before the answer:

    python chat.py -c CollectionName -d ./db -e "all-MiniLM-L6-v2" -m llama --variant "3.2-iB" --context

Show all the options of the script:

    python chat.py -h

Full list of tested models:

* Google/gemma-3-1b-it
* Google/gemma-3-4b-it
* Google/gemma-3-12b-it
* Google/gemma-3-1b-it-qat-q4_0
* Google/gemma-3-4b-it-qat-q4_0
* Google/gemma-3-12b-it-qat-q4_0
* Google/gemma-3-27b-it-qat-q4_0
* Meta-llama/llama-3.1-8B-Instruct
* Meta-llama/llama-3.2-1B-Instruct
* Meta-llama/llama-3.2-3B-Instruct
* Qwen/qwen3-0.6B
* Qwen/qwen3-1.7B
* Qwen/qwen3-4B
* Qwen/qwen3-8B
* Mistralai/mistral-7b-instruct-v0.3

# Test cripts

This repository provides several scripts to perform isolated steps of the process, this is usefull for testing and to understand the process better.

* **extract_info.py**: Script to test the text and OCR information extraction.
* **get_embeddings.py**: Script to test the extraction of the structure and embeddings of a file previously processed with *extract_info.py*.
* **get_relevant.py**: Script to test the query capabilities of CromaDB storage by retrieving relevant documents given a query sentence.

## extract_info.py

Script to extract information from PDF files, the output can be plain text or CSV data with information of the text and layout of the file.

This script loads a PDF file (or multiple files in a directory) and process them:

1. Load text from the PDF.
2. Use Tesseract to extract OCR information from the file.
3. Combine plain text and OCR information.
4. Output one of the following:
    * Plain text (.txt)
    * Layot information (OCR) as obtained from Tesseract (.csv)
    * Layout information (OCR) combined with plain text (main feature of the script) (.csv).
5. The layout information or the plain text can be passed to the *get_embeddings.py* script to continue the process.

### Examples

Process a PDF file and show its text in console:

    python extract_info.py -f /path/to/file.pdf

Process a PDF file and save the output text in a file:

    python extract_info.py -f /path/to/file.pdf -o /path/to/out.txt

Process a single page of a PDF file:

    python extract_info.py -f /path/to/file.pdf -p <page>

Process a PDF file and save the data as CSV for latter processing. The output csv can be passed to *get_embeddings.py* script:

    python extract_info.py -f /path/to/file.pdf -t csv -o /path/to/out.csv

Process all the PDF files in a directory and save a .txt file for each PDF file:

    python extract_info.py -d /path/to/dir/ -o /path/to/out/

Process all the PDF files in a directory and save a .csv file for each PDF file. The output csv files can be passed to *get_embeddings.py* script:

    python extract_info.py -d /path/to/dir/ -t csv -o /path/to/out/

Keep cache of the file, usefull when the same file is going to be processed multiple times. *-k* option is compatible with all the other options:

    python extract_info.py -f /path/to/file.pdf -t csv -o /path/to/out.csv -k

Show all the options of the script:

    python extract_info.py -h

## get_embeddings.py

Script that loads the CSV data obtained with *extract_info.py* to obtain the structure of the file, calculate the embeddings and store the in a database. The database is built using ChromaDB.

This script loads a CSV file (or multiple files in a directory) and process them:

1. Load data as csv or plain text.
2. When using data as input, the script identifies titles and sections by its possition and by its content, using regular expressions.
3. When using plain text as input, the script identifies titles and sections by its contents only, using regular expressions.
4. Use the identified sections to build a tree representation of the file.
5. From each node of the tree, calculate the embeddings of its contents.
6. Obtain metadata of the node (e.g: Path, title, etc.)
7. Create a collection in the database (or use an existing one) and add the embeddings and its metadata.

### Examples

Load CSV data and show the tree of the document as an image:

    python get_embeddings.py -f /path/to/file.csv -a tree

Load CSV data and show the tree structure of the document as a string:

    python get_embeddings.py -f /path/to/file.csv -a structure

Load CSV data and save the tree of the document as an image:

    python get_embeddings.py -f /path/to/file.csv -a tree -o /path/to/out.png

Load CSV data, calculate the embeddings of each section and save them in a collection
using ChromaDB. Database is created by default in *db/*:

    python get_embeddings.py -f /path/to/file.csv -a embeddings -e "all-MiniLM-L6-v2" -c <CollectionName> -s chromadb --inner-splitter section

Load CSV data, calculate the embeddings of each paragraph and save them in a collection
using ChromaDB. Database is created by default in *db/*:

    python get_embeddings.py -f /path/to/file.csv -a embeddings -e "all-MiniLM-L6-v2" -c <CollectionName> -s chromadb --inner-splitter paragraph

Load TXT and show the tree of the document as an image:

    python get_embeddings.py -f /path/to/file.txt -t txt -a tree

Load TXT, calculate the embeddings of each section and save them in a collection
using ChromaDB. Database is created by default in *db/*:

    python get_embeddings.py -f /path/to/file.txt -t txt -a embeddings -e "all-MiniLM-L6-v2" -c <CollectionName> -s chromadb --inner-splitter section

Show all the options of the script:

    python get_embeddings.py -h

## get_relevant.py

Once the database is created with *get_embeddings.py* this script recieves a sentence or question and searches in the database for related documents using cosine similarity.

1. Convert query string to an embedding (with ChromaDB).
2. Query the collection for similar documents.
3. Show the textual content and some metadata from the related documents.

### Examples

Retrieve the 5 most relevant documents for a query:

    python get_relevant.py -c <CollectionName> -e "all-MiniLM-L6-v2" -n 5 "Que tipos de profesores hay?"

Show all the options of the script:

    python get_relevant.py -h

# Library

In case you want to use the modules directly from code here is a full example usage
to load PDF file and combine plain text and OCR information to get the embeddings:

    from document_loaders.pdf import PyPDFMixedLoader
    from document_splitters.hierarchical import DataTreeSplitter
    from llms.storage import ChromaDBStorage

    # Load file and merge all information
    pdf_loader = PyPDFMixedLoader('./.cache', keep_cache=False)
    pdf_loader.load("/path/to/file.pdf")

    # Get document data as PdfDocumentData
    data = pdf_loader.get_document_data()

    # Create a tree with the structure of the document
    splitter = DataTreeSplitter(data.get_data(remove_headers=True), "nombre_documento")
    splitter.analyze()

    # Split document in sentences
    sentences = []
    metadatas = []
    documents = splitter.extract_documents(inner_splitter='section')
    for doc in documents:
        sentences.append(doc['content'])
        metadatas.append(doc['metadata'])

    # Store embeddings
    storage = ChromaDBStorage('all-MiniLM-L6-v2', './db')
    storage.save_info("CollectionName", {
        'sentences': sentences,
        'metadatas': metadatas
    })

