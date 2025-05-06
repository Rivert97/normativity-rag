from typing import List
import argparse
import os
import dotenv

from AppLogger import AppLogger
from PdfDocumentData import PdfDocumentData
from Splitters import NormativitySplitter
from Embedders import AllMiniLM
from Storage import CSVStorage

dotenv.load_dotenv()

PROGRAM_NAME = 'EmbeddingsCLI'
VERSION = '1.00.00'

EMBEDDERS = {
    'AllMiniLM': AllMiniLM,
}

class CLIException(Exception):
    def __init__(self, message):
        super().__init__(f"{PROGRAM_NAME} ERROR: {message}")

class CLIController():
    """This class controls the execution of the program when using
    CLI.
    """
    def __init__(self):
        self._logger = AppLogger.get_logger('CLIController')

        self.print_to_console = True
        self.embedder = None

        self._args = self.__process_args()

    def run(self):
        if self._args.type == 'csv':
            self.__process_csv_file(self._args.file)
        elif self._args.type == 'txt':
            self.__process_txt_file(self._args.file)
        else:
            raise CLIException("Tipo de archivo no valido")

    def __process_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            prog=PROGRAM_NAME,
            description='Creates a vectorized database of data extracted from PDF files',
            epilog=f'%(prog)s-{VERSION}, Roberto Garcia <r.garciaguzman@ugto.mx>'
        )

        parser.add_argument('-a', '--action', default='embeddings', choices=['embeddings', 'structure', 'tree'], type=str, help='Action to perform: "embeddings" to split the file and get the embeddings, "structure" to show file structure in console. "tree" to show an image of the tree of titles of the file.')
        parser.add_argument('-e', '--embedder', default='AllMiniLM', choices=['AllMiniLM'], type=str, help='Embeddings function to be used')
        parser.add_argument('-f', '--file', default='', type=str, help='Path to file containing the data or text of the document')
        parser.add_argument('-o', '--output', default='', help='Name of the file to be saved')
        parser.add_argument('-p', '--page', type=int, help='Number of page to be processed')
        parser.add_argument('-t', '--type', default='csv', choices=['csv', 'txt'], type=str, help='Type of input')
        parser.add_argument('--version', action='store_true', help='Show version of this tool')

        args = parser.parse_args()

        if args.action == 'tree' and args.output == '':
            raise CLIException("Please specify an output for the tree image")

        if args.file != '' and not os.path.exists(args.file):
            raise CLIException(f"File '{args.file} not found")

        if args.output != '':
            basedir = os.path.dirname(args.output)
            if basedir != '' and not os.path.exists(basedir):
                raise CLIException(f"Output path '{args.output}' does not exist")

        if args.file == '':
            raise CLIException(f"Please specify an input file")

        if args.output != '':
            self.print_to_console = False

        if args.embedder in EMBEDDERS:
            self.embedder = EMBEDDERS[args.embedder]
        else:
            raise CLIException(f"Invalid embedder '{args.embedder}'")

        return args

    def __process_csv_file(self, filename: str):
        if self._args.action == 'embeddings':
            self.__action_embeddings(filename)
        elif self._args.action == 'structure':
            self.__action_structure(filename)
        elif self._args.action == 'tree':
            self.__action_tree(filename)
        else:
            raise CLIException("Invalid action '{self._args.action}'")

    def __process_txt_file(self, filename: str):
        raise CLIException("Functionality not implemented")

    def __action_embeddings(self, filename:str):
        splitter = self.__load_and_split_doc(filename)
        embed = self.embedder()

        sentences = []
        metadatas = []
        documents =splitter.extract_documents()
        for doc in documents:
            sentences.append(doc.get_content())
            metadatas.append(doc.get_metadata())

        embeddings = embed.get_embeddings(sentences)

        if self.print_to_console:
            print('sentences,metadatas,embeddings')
            for sent, meta, emb in zip(sentences, metadatas, embeddings):
                print(f'{sent},{meta},{emb}')
        else:
            storage = CSVStorage()
            storage.save_info(self._args.output, sentences, metadatas, embeddings)

    def __action_structure(self, filename:str):
        splitter = self.__load_and_split_doc(filename)
        if self.print_to_console:
            splitter.show_file_structure()
        else:
            structure = splitter.get_file_structure()
            self.__save_txt_file(self._args.output, structure)

    def __action_tree(self, filename:str):
        splitter = self.__load_and_split_doc(filename)
        splitter.show_tree(self._args.output)

    def __load_and_split_doc(self, filename:str):
        document_data = PdfDocumentData()
        document_data.load_data(filename)
        if self._args.page != None:
            splitter = NormativitySplitter(document_data.get_page_data(self._args.page, remove_headers=True))
        else:
            splitter = NormativitySplitter(document_data.get_data(remove_headers=True))

        splitter.analyze()

        return splitter

    def __calculate_embeddings(self, sentences:List[str]):
        embed = self.embedder()

        return embed.get_embeddings(sentences)

    def __save_txt_file(self, filename:str, content:str):
        with open(filename, 'w') as f:
            f.write(content)

if __name__ == "__main__":
    try:
        AppLogger.setup_root_logger(
            int(os.getenv('LOG_LEVEL', '20')),
            os.getenv('LOG_FILE', f"{PROGRAM_NAME}.log"),
            int(os.getenv('LOG_CONSOLE', '0'))
        )
    except ValueError as e:
        print(f"Environment variables file is corrupted: {e}")
        exit(1)

    _logger = AppLogger.get_logger(PROGRAM_NAME)

    try:
        controller = CLIController()
        controller.run()
    except CLIException as e:
        print(e)
        _logger.error(e)
        exit(1)