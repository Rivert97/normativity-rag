"""Script to manage ChromaDB collections."""

import os
import argparse
import chromadb
import chromadb.errors

from .utils.controllers import CLI, run_cli
from .utils.exceptions import CLIException
from .utils.defaults import Defaults

PROGRAM_NAME = 'ManageChromaCLI'
VERSION = '1.00.00'

class ManageChromaCLI(CLI):
    """This class controls the execution of the program when using
    CLI.
    """
    def __init__(self):
        super().__init__(PROGRAM_NAME, __doc__, VERSION)

        self._args = None

    def run(self):
        """Run the script logic."""
        self._logger.debug('Loading database')
        client = chromadb.PersistentClient(path=self._args.database_dir)

        if self._args.action == "list":
            list_collections(client)
        elif self._args.action == "delete" and self._args.file != "":
            delete_file_from_collection(client, self._args.collection, self._args.file)
        elif self._args.action == "delete" and self._args.file == "":
            delete_collection(client, self._args.collection)

    def process_args(self) -> argparse.Namespace:
        super().process_args()

        self.parser.add_argument("-a", "--action",
                                 choices=["list", "delete"],
                                 required=True,
                                 help="Action to perform.")
        self.parser.add_argument("-c", "--collection",
                                 help="Name of the collection.")
        self.parser.add_argument("-d", "--database-dir",
                                 default=Defaults.database_dir,
                                 help=f"""
                                    Path to the ChromaDB database.
                                    Default: {Defaults.database_dir}.
                                    """)
        self.parser.add_argument("-f", "--file",
                                 default="",
                                 help="Name of the file to modify.")

        args = self.parser.parse_args()

        if args.action != "list" and not args.collection:
            raise CLIException("Collection is required when action is not list")

        if not os.path.exists(args.database_dir):
            raise CLIException("Database directory does not exist")

        self._args = args

def list_collections(client):
    """List all collections in the database."""
    try:
        collections = client.list_collections()
        if not collections:
            print("No collections found.")
            return

        print("Collections:")
        for collection in collections:
            print(f"- {collection.name}")
    except chromadb.errors.ChromaError as e:
        print(f"Error listing collections: {e}")

def delete_collection(client, collection_name):
    """Delete a collection from the database."""
    try:
        client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' deleted successfully.")
    except chromadb.errors.ChromaError as e:
        print(f"Error deleting collection '{collection_name}': {e}")

def delete_file_from_collection(client, collection_name, file_name):
    """Delete a file from a collection."""
    try:
        collection = client.get_collection(name=collection_name)
        collection.delete(
            where={"document_name": file_name.rstrip('.pdf')}
        )
        print(f"File '{file_name}' deleted successfully from collection '{collection_name}'.")
    except chromadb.errors.ChromaError as e:
        print(f"Error deleting file '{file_name}' from collection '{collection_name}': {e}")

def main():
    """Run the script."""
    run_cli(ManageChromaCLI)

if __name__ == "__main__":
    run_cli(ManageChromaCLI)
