# Using the chat

The chat script allows to send messages to a model that performs RAG from a previously created database with *run.py extract*.

The script can be used in interactive mode to answer multiple questions in sequence or in single-query mode.

The process followed by the script is as follows:

1. Load the collection of embeddings.
2. Convert the query to embedding and search for similar embeddings in the database to obtain related documents.
3. Load an instruction-following model from HuggingFace, currently available: Gemma, Llama, Qwen, Mistral.
3. Pass the retrieved documents and its metadata as context to the model, alongside the query.
4. Let the model generate the answer and show it to the user.

Full list of options of the script can be obtained using the -h option.

    python run.py chat -h

## Examples

Single-query mode, get the answer to a simple question:

    python run.py chat -c CollectionName -d ./db -e "all-MiniLM-L6-v2" -m Qwen/Qwen3-0.6B --query "Que tipos de profesores hay?"

> __NOTE:__ To know the id for the model visit HuggingFace models collection (https://huggingface.co/models). Consider that not al models are available.

Interactive mode:

    python run.py chat -c CollectionName -d ./db -e "all-MiniLM-L6-v2" -m google/gemma-3-1b-it

Interactive mode with context. This option shows the relevant documents before the answer:

    python run.py chat -c CollectionName -d ./db -e "all-MiniLM-L6-v2" -m meta-llama/Llama-3.2-1B-Instruct --context

Full list of tested models:

* Google/gemma-3-1b-it
* Google/gemma-3-4b-it
* Google/gemma-3-12b-it
* Google/gemma-3-1b-it-qat-q4_0
* Google/gemma-3-4b-it-qat-q4_0
* Google/gemma-3-12b-it-qat-q4_0
* Google/gemma-3-27b-it-qat-q4_0
* Meta-llama/llama-3.1-8B-Instruct
* Meta-llama/llama-3.2-1B-Instruct
* Meta-llama/llama-3.2-3B-Instruct
* Qwen/qwen3-0.6B
* Qwen/qwen3-1.7B
* Qwen/qwen3-4B
* Qwen/qwen3-8B
* Mistralai/mistral-7b-instruct-v0.3
