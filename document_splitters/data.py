"""Module to define classes to store different types of documents."""

class Document:
    """Class to store the information of a chunk of text."""

    def __init__(self, content:str, metadata:dict=None):
        self.content = content
        self.metadata = metadata

    def get_content(self):
        """Get the text contained in the document."""
        return self.content

    def get_metadata(self):
        """Get the metadata associated with the document."""
        return self.metadata

    def __str__(self):
        if self.metadata:
            metadata_str = ', '.join([f"{key}:{value}" for key, value in self.metadata.items()])
        else:
            metadata_str = 'None'
        return f"Document(content='{self.content}', metadata={{{metadata_str}}})"
