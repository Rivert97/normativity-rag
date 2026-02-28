# Linux installation guide

### Requirements

* Python 3.12.7

### Update pip

    pip install --upgrade pip


## Using AWS Bedrock

Follow this steps if you want to use AWS Bedrock models.

### Install python packages

Install only the basic requirements:

    pip install -r requirements.bedrock.txt

## Using HuggingFace models (withouth .gguf models)

Follow this steps if you want to use HuggingFace models. This option does not support models in .gguf format.

### Install python packages

Install the full requirements:

    pip install -r requirements.txt

> __NOTE:__ If an error occurs during the installation of `llama_cpp_python`, remove the entry from the `requirements.txt` file and try again.

## Using HuggingFace models (with .gguf models)

The easiest way to use models in .gguf format is using a Docker container, since the llama_cpp_python library is not easy to instal in Windows.

### Docker

1. Install Docker Desktop for Windows
2. Run the following command to pull the image:

    docker pull Rivert97/normativity-rag:latest

    > __NOTE:__ You can also build your own image using the Dockerfile in the root directory. First create a llama_cpp image with the `Dockerfile.llama` and then the normativity-rag image with the `Dockerfile`.

See instructions on how to use the Docker image in the [Docker](../docker/docker.md) section.