# Simple RAG for Norm-like files

This is a project that implements a simple RAG system. It is specially adapted to process normativity documents, this means, files with articles, sections and/or titles.

The full project performs the basic RAG steps:

* **PDF document loading:** This project can load either only the plain text or the text+position inside the document. Obtaining the text+position allows the program to detect sections and titles in a better way.
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

# Configuration

To configure the application, you need to set up the environment variables. A template file `.env.example` is provided.

1. Copy the example file to `.env`:

   ```bash
   cp .env.example .env
   ```

2. Open the `.env` file and update the variables as needed:

   * **Logging:**
     * `LOG_LEVEL`: Set the logging level (e.g., 10 for DEBUG, 20 for INFO).
     * `LOG_FILE`: Path to the log file.
     * `LOG_CONSOLE`: Set to 1 to enable console logging.

   * **HuggingFace (Optional):**
     * `HF_TOKEN`: Your HuggingFace API token. Required for accessing gated models.
     * `HF_HOME`: Directory to store downloaded models.

   * **Model Settings:**
     * `EMBEDDING_CONTEXT`: Context size for embeddings.
     * `MODEL_CONTEXT`: Context size for the LLM.

   * **AWS Bedrock (Optional):**
     * `AWS_ACCESS_KEY_ID`: Your AWS Access Key.
     * `AWS_SECRET_ACCESS_KEY`: Your AWS Secret Key.
     * `AWS_REGION`: AWS Region (e.g., `us-east-1`).

    > __NOTE:__ When using Bedrock models, the model ID needs to be prefixed with 'bedrock/' in the `run.yml` file and in the --model or --embedder options.

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

* [Extracting information from PDF file](./docs/scripts/extract_info.md): Script to test the text and visual information extraction.
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

