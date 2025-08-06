"""Script to execute multiple functions and sub-scripts.

usage:python run.py <command> [OPTIONS]

command:
  chat                  Open a chat with or without RAG
  eval_rag              Evaluate the full RAG system
  eval_retrieval        Evaluate the retrieval part of the system
  extract_info          Extrar text or information from files
  extract               Run all the process of extraction, embeddings creation and storage
  get_embeddings        Get the embeddings from previously extracted data
  get_relevant          Get relevant documents to a question from a previously made database

options:
  <command> -h          For more information on each command
"""
import sys
import importlib

def main():
    """Run the script."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "-h":
        print(__doc__)
        sys.exit(0)

    try:
        # Try to import the script dynamically
        module = importlib.import_module(f"scripts.{command}")
    except ModuleNotFoundError:
        print(f"Unknown command: {command}")
        sys.exit(1)

    sys.argv = [f"run.py {command}"] + args  # Fix argv for argparse inside the script
    module.main()

if __name__ == "__main__":
    main()
