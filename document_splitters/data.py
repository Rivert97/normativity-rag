class Document:

    def __init__(self, content:str, metadata:dict={}):
        self.content = content
        self.metadata = metadata

    def get_content(self):
        return self.content

    def get_metadata(self):
        return self.metadata

    def __str__(self):
        return f"Document(content='{self.content}', metadata={{{', '.join([f"{key}:{value}" for key, value in self.metadata.items()])}}})"