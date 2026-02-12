# Information extractor

This script processes the PDF files in two stages: Extracts text and visual information from the file and gets the structure of the document and its embeddings.

The script recieves a PDF file or directory of files as an input and creates a collection of embeddings in a database.

The script can be used passing all the options as paramters to the script or with a configuration file in YAML format.

Full list of options of the script can be obtained using the -h option.

    python run.py extract -h

## Examples

Process a PDF file and save the embeddings in a collection:

    python run.py extract -c <CollectionName> -e "all-MiniLM-L6-v2" --extraction-type data -f /path/to/file.pdf --inner-splitter section

Process a directory and save the embeddings in a collection from a settings file:

    python run.py extract --settings-file /path/to/settings.yml

## YALM config example

    db:
        directory: /path/to/dir/

        settings:
            database_dir: ./db

        collections:
            CollectionName:
                embedder: all-MiniLM-L6-v2
                extraction_type: data
                inner_splitter: section
