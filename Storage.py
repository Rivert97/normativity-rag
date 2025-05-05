from typing import List
import chromadb

from Splitters import Document

class Storage:

    def __init__(self):
        self.client = chromadb.PersistentClient(path="./db")

    def save_documents(self, name:str, documents:List[Document]):
        try:
            collection = self.client.get_collection(name)
            print("Document already exists")
            return
        except chromadb.errors.NotFoundError as e:
            collection = self.client.create_collection(name)

        collection.add(
            documents=[d.get_content() for d in documents],
            metadatas=[d.get_metadata() for d in documents],
            ids = [str(i+1) for i in range(len(documents))]
        )
