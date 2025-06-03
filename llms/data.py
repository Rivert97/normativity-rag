"""Module to define classes to store different types of documents."""
import numpy as np

class Document:
    """Class to store the information of a chunk of text."""

    def __init__(self, content:str, metadata:dict=None, embeddings:np.array=None):
        self.content = content
        self.metadata = metadata
        self.embeddings = embeddings

    def get_content(self):
        """Get the text contained in the document."""
        return self.content

    def get_metadata(self):
        """Get the metadata associated with the document."""
        return self.metadata

    def get_embeddings(self):
        """Get the embeddings array."""
        return self.embeddings

    def get_reference(self):
        """Return a string with a human readable reference to the document."""
        ref = self.metadata['document_name']
        ref += ', '.join(self.metadata['path'].strip("'").split('/')[2:])

        return ref

    def get_for_standard_pipeline(self):
        """Return a dictionary formated for Hugging Face standard 'rag' pipeline."""
        doc = {
            'title': self.metadata['title'],
            'text': self.content,
        }

        return doc

    def __str__(self):
        if self.metadata:
            metadata_str = ', '.join([f"{key}:{value}" for key, value in self.metadata.items()])
        else:
            metadata_str = 'None'
        return f"Document(content='{self.content}', metadata={{{metadata_str}}})"
