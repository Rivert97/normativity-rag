# Information extractor

This script processes the PDF files in two stages: Extracts text and visual information from the file and gets the structure of the document and its embeddings.

The script recieves a PDF file or directory of files as an input and creates a collection of embeddings in a database.

The script can be used passing all the options as paramters to the script or with a configuration file in YAML format.

Full list of options of the script can be obtained using the -h option.

    python run.py extract -h

## Examples

Process a PDF file and save the embeddings in a collection. Use *-k* to keep cache when processing the same file multiple times:

    python run.py extract -c <CollectionName> -e "all-MiniLM-L6-v2" --loader pdfplumber --extraction-type data -f /path/to/file.pdf --inner-splitter section -k

Process a directory and save the embeddings in a collection from a settings file:

    python run.py extract --settings-file /path/to/settings.yml

## YALM config example

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
