# Linux installation guide

## Requirements

* Python 3.12.7

This guide was made considering Ubuntu-based distros, and tested in Kubuntu 24.04.

## Download repository

    git clone https://github.com/Rivert97/normativity-rag.git

## Install Ubuntu packages

Poppler-Utils must be installed in the system:

    sudo apt-get install poppler-utils

> NOTE: Next step is only needed if you want to use pyTesseract.

Tesseract must be installed in the system. Follow instructions in source repo:
[tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file).

## Update pip

    pip install --upgrade pip

## Install python packages

Install all the packages from *requirements.txt*:

    pip install -r requirements.txt
