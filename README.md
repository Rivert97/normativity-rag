# Simple RAG for Norm-like files

This is a project that implements a simple RAG system. It is specially adapted to process normativity documents, this means, files with articles, sections and/or titles.

The full project performs the basic RAG steps:

* **PDF document loading:** This project can load either only the plain text or the text+position inside the document. Obtaining the text+position allows the program to detect sections and titles in a better way. It can be performed in different ways:
    * Load plain text only with pypdf, pdfplumber or pytesseract (OCR).
    * Load text+position combining plain text and OCR.
    * Load text+position using pdfplumber.
* **PDF document splitting:** Creates a tree representation of the document, where each node corresponds to an article or section. It uses positional data and regular expressions to find the different sections. Each section can be
subdivided in one or more chunks.
* **Embeddings creation:** Uses an embeddings LLM to obtain the vector representations of the chunks. Multiple models available.
* **Embeddings storage:** Creates a ChromaDB database containing the embeddings and the metadata of each chunk.
* **Chat:** Shows an interactive console where the user can ask questions
about the documents and receive the referenced answers.

# Installation

Please refer to the proper installation guide according to your system.

* Linux: [Linux Installation Guide](./docs/install/linux.md)
* Windows: Currently not supported
* MaxOS: N/A

# Quick Start

To follow the guide you will need to create the directory *documents/* where the source PDF files should be moved.

```
.
├── docs
├── documents <-- Add this folder
├── README.md
├── requirements.txt
├── run.py
├── run.yml.example
├── scripts
├── simplerag
└── tests
```

Now we create the database from all the documents in the folder:

    python run.py extract -c CUSTOM_COLLECTION -d ./documents

> __NOTE:__ The first time running the script it will download the model to create the embeddings.

Once the database was created, we can initiate a chat with an LLM model
and it will answer the questions regarding the documents.

    python run.py chat --show-context -c CUSTOM_COLLECTION

An interactive console is opened where you can ask questions.

For a detailed documentation on each functionality, please go to the corresponding doc file.

* [Extractor full documentation](./docs/scripts/extract.md)
* [Using the chat](./docs/script/chat.md)

# Step-by-step scripts

In order to allow a better debugging of the processes or to obtain some extra resources, the project provides several script to perform isolated steps of the full process, this is usefull for testing and to understand the process better.

* [Extracting information from PDF file](./docs/scripts/extract_info.md): Script to test the text and OCR information extraction.
* [Getting the embeddings from the information](./docs/scripts/get_embeddings.md): Script to test the extraction of the structure and embeddings of a file previously processed with *extract_info.py*.
* [Retrieving relevant documents from the database](./docs/scripts/get_relevant.md): Script to test the query capabilities of CromaDB storage by retrieving relevant documents given a query sentence.

# Library

In case you want to use the modules directly from code here is a full example usage to load PDF file with PdfPlumber to get the embeddings:

    from document_loaders.pdf import PDFPlumberLoader
    from document_splitters.hierarchical import DataTreeSplitter
    from llms.storage import ChromaDBStorage

    # Load file and merge all information
    pdf_loader = PDFPlumberLoader("/path/to/file.pdf")

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
    },
    id_prefix='file_001')

