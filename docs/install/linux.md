# Linux installation guide

## Requirements

* Python 3.12.7

This guide was made considering Ubuntu-based distros, and tested in Kubuntu 24.04.

## Install Ubuntu packages

Poppler-Utils must be installed in the system:

    sudo apt-get install poppler-utils

Tesseract must be installed in the system. Follow instructions in source repo:
[tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file).

## Update pip

    pip install --upgrade pip

## Install PyTorch

Follow installation guide to install PyTorch according to your CUDA compatible GPU.
[PyTorch Get Started](https://pytorch.org/get-started/locally/).

This repo is tested on:

* Pytorch: 2.7.1
* Using pip
* CUDA 11.8

It's not necessary to install torchaudio:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

## Install python packages

Install all the packages from *requirements.txt*:

    pip install -r requirements.txt
