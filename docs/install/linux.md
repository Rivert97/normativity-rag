# Linux installation guide

## Requirements

* Python 3.12.7

This guide was made considering Ubuntu-based distros, and tested in Kubuntu 24.04.

## Download repository

    git clone https://github.com/Rivert97/normativity-rag.git

## Install Ubuntu packages

Poppler-Utils must be installed in the system:

    sudo apt-get install poppler-utils

## Install Llama-cpp-python (Optional)

In case you want to use local models in .gguf format, you need to install llama-cpp.

Check the official documentation for more information: https://github.com/abetlen/llama-cpp-python

## Update pip

    pip install --upgrade pip

## Install python packages

If you want a lightweight version that uses AWS Bedrock for calculating the embeddings and inference, install only the basic requirements:

    pip install -r requirements.bedrock.txt

If you want to use local models downloaded from HuggingFace, install the full requirements:

    pip install -r requirements.txt
