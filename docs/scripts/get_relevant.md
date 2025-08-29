# Retrieven relevant documents from the database

Once the database is created with *run.py get_embeddings* this script recieves a sentence or question and searches in the database for related documents using cosine similarity.

1. Convert query string to an embedding (with ChromaDB).
2. Query the collection for similar documents.
3. Show the textual content and some metadata from the related documents.

Full list of options of the script can be obtained using the -h option.

    python run.py get_relevant -h

## Examples

Retrieve the 5 most relevant documents for a query:

    python run.py get_relevant -c <CollectionName> -e "all-MiniLM-L6-v2" -n 5 "Que tipos de profesores hay?"
