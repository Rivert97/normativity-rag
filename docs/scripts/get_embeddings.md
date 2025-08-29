# Getting the embeddings from the information

Script that loads the CSV data obtained with *run.py extract_info* to obtain the structure of the file, calculate the embeddings and store the in a database. The database is built using ChromaDB.

This script loads a CSV file (or multiple files in a directory) and process them:

1. Load data as csv or plain text.
2. When using data as input, the script identifies titles and sections by its possition and by its content, using regular expressions.
3. When using plain text as input, the script identifies titles and sections by its contents only, using regular expressions.
4. Use the identified sections to build a tree representation of the file.
5. From each node of the tree, calculate the embeddings of its contents.
6. Obtain metadata of the node (e.g: Path, title, etc.)
7. Create a collection in the database (or use an existing one) and add the embeddings and its metadata.

Full list of options of the script can be obtained using the -h option.

    python run.py get_embeddings -h

## Examples

Load CSV data and show the tree of the document as an image:

    python run.py get_embeddings -f /path/to/file.csv -a tree

Load CSV data and show the tree structure of the document as a string:

    python run.py get_embeddings -f /path/to/file.csv -a structure

Load CSV data and save the tree of the document as an image:

    python run.py get_embeddings -f /path/to/file.csv -a tree -o /path/to/out.png

Load CSV data, calculate the embeddings of each section and save them in a collection
using ChromaDB. Database is created by default in *db/*:

    python run.py get_embeddings -f /path/to/file.csv -a embeddings -e "all-MiniLM-L6-v2" -c <CollectionName> -s chromadb --inner-splitter section

Load CSV data, calculate the embeddings of each paragraph and save them in a collection
using ChromaDB. Database is created by default in *db/*:

    python run.py get_embeddings -f /path/to/file.csv -a embeddings -e "all-MiniLM-L6-v2" -c <CollectionName> -s chromadb --inner-splitter paragraph

Load TXT and show the tree of the document as an image:

    python run.py get_embeddings -f /path/to/file.txt -t txt -a tree

Load TXT, calculate the embeddings of each section and save them in a collection
using ChromaDB. Database is created by default in *db/*:

    python run.py get_embeddings -f /path/to/file.txt -t txt -a embeddings -e "all-MiniLM-L6-v2" -c <CollectionName> -s chromadb --inner-splitter section
