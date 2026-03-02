# Simple RAG for Norm-like files

This is a project that implements a simple RAG system. It is specially adapted to process normativity documents, this means, files with articles, sections and/or titles.

The full project performs the basic RAG steps:

* **PDF document loading:** This project can load either only the plain text or the text+position inside the document. Obtaining the text+position allows the program to detect sections and titles in a better way.
* **PDF document splitting:** Creates a tree representation of the document, where each node corresponds to an article or section. It uses positional data and regular expressions to find the different sections. Each section can be subdivided in one or more chunks.
* **Embeddings creation:** Uses an embeddings LLM to obtain the vector representations of the chunks. Multiple models available.
* **Embeddings storage:** Creates a ChromaDB database containing the embeddings and the metadata of each chunk.
* **Chat:** Shows an interactive console where the user can ask questions
about the documents and receive the referenced answers.

# Installation

Please refer to the proper installation guide according to your system.

* Linux: [Linux Installation Guide](./docs/install/linux.md)
* Windows: [Windows Installation Guide](./docs/install/windows.md)
* MaxOS: N/A
* Docker: [Docker Installation Guide](./docs/docker/docker.md)

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

   * **Model Settings:**
     * `EMBEDDING_CONTEXT`: Context size for embeddings.
     * `MODEL_CONTEXT`: Context size for the LLM.

   * **AWS Bedrock (Optional, only if using AWS):**
     * `AWS_ACCESS_KEY_ID`: Your AWS Access Key ID.
     * `AWS_SECRET_ACCESS_KEY`: Your AWS Secret Access Key.
     * `AWS_REGION`: AWS Region (e.g., `us-east-1`).

    > __NOTE:__ You can add any additional environment variable according to your need (e.g., `HF_TOKEN`).

# Quick Start (Using Local models)

1. Create a directory with the documents you want to process:

    ```bash
    mkdir /home/$USER/documents
    ```

2. Create the database from all the documents in the directory:

    ```bash
    python run.py extract -c CUSTOM_COLLECTION -d /home/$USER/documents
    ```

  > __NOTE:__ The default model is `all-MiniLM-L6-v2`, the first time running the script it will download the model to create the embeddings.

3. Once the database was created, we can initiate a chat with an LLM model and it will answer the questions regarding the documents.

    ```bash
    python run.py chat --show-context -c CUSTOM_COLLECTION
    ```

# Quick Start (Using AWS Bedrock models)

1. Setup your AWS account with an IAM user with access to the needed models. Get the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` from the IAM user.

2. Create a directory with the documents you want to process:

    ```bash
    mkdir /home/$USER/documents
    ```

3. Create the database from all the documents in the directory:

    ```bash
    python run.py extract -c CUSTOM_COLLECTION -d /home/$USER/documents -e bedrock/amazon.titan-embed-text-v2:0
    ```

  > __NOTE:__ AWS model IDs are prefixed with `bedrock/`.

4. Once the database was created, we can initiate a chat with an LLM model and it will answer the questions regarding the documents.

    ```bash
    python run.py chat --show-context -c CUSTOM_COLLECTION -e bedrock/amazon.titan-embed-text-v2:0 -m bedrock/openai.gpt-oss-20b-1:0
    ```

# Script documentation

For a detailed documentation on each functionality, please go to the corresponding doc file.

* [Extractor full documentation](./docs/scripts/extract.md)
* [Using the chat](./docs/script/chat.md)

# Step-by-step scripts

In order to allow a better debugging of the processes or to obtain some extra resources, the project provides several script to perform isolated steps of the full process, this is usefull for testing and to understand the process better.

* [Extracting information from PDF file](./docs/scripts/extract_info.md): Script to test the text and visual information extraction.
* [Getting the embeddings from the information](./docs/scripts/get_embeddings.md): Script to test the extraction of the structure and embeddings of a file previously processed with *extract_info.py*.
* [Retrieving relevant documents from the database](./docs/scripts/get_relevant.md): Script to test the query capabilities of CromaDB storage by retrieving relevant documents given a query sentence.

# Library

In case you want to use the modules directly from code here is a full example usage to load PDF and get the embeddings:

    from simplerag.document_loaders.pdf import PDFPlumberLoader
    from simplerag.document_splitters.hierarchical import DataTreeSplitter
    from simplerag.llms.storage import ChromaDBStorage
    from simplerag.llms.embedders import EmbedderParams

    # Load file and merge all information
    pdf_loader = PDFPlumberLoader("/path/to/file.pdf")

    # Get document data as PdfDocumentData
    data = pdf_loader.get_document_data()

    # Create a tree with the structure of the document
    boundaries = {
        'top': 0.1,
        'bottom': 0.95,
        'left': 0.05,
        'right': 0.95,
    }
    splitter = DataTreeSplitter(
      data.get_data(remove_headers=True, boundaries=boundaries),
      "nombre_documento",
      DataSplitterOptions(max_characters=8000)
    )
    splitter.analyze()

    # Split document in sentences
    sentences = []
    metadatas = []
    documents = splitter.extract_documents(inner_splitter='section')
    for doc in documents:
        sentences.append(doc['content'])
        metadatas.append(doc['metadata'])

    # Store embeddings
    embedder_params = EmbedderParams(embedding_context=2048)
    storage = ChromaDBStorage('all-MiniLM-L6-v2', './db', embedder_params=embedder_params)
    storage.save_info("CollectionName", {
        'sentences': sentences,
        'metadatas': metadatas
    },
    id_prefix='file_001')

