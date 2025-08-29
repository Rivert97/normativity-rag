# Extracting information from PDF file

Script to extract information from PDF files, the output can be plain text or CSV data with information of the text and layout of the file.

This script loads a PDF file (or multiple files in a directory) and process them:

1. Load text from the PDF.
2. (Optional) Use Tesseract to extract OCR information from the file.
3. (Optional) Combine plain text and OCR information.
4. Output one of the following:
    * Plain text (.txt)
    * Layot information obtained from Tesseract (.csv)
    * Layout information combining plain text and tesseract (main feature of the script) (.csv).
    * Layout information using Pdfplumber
5. The layout information or the plain text can be passed to the *run.py get_embeddings* script to continue the process.

Full list of options of the script can be obtained using the -h option.

    python run.py extract_info -h

## Examples

Process a PDF file and show its text in console:

    python run.py extract_info -f /path/to/file.pdf

Process a PDF file and save the output text in a file:

    python run.py extract_info -f /path/to/file.pdf -o /path/to/out.txt

Process a single page of a PDF file:

    python run.py extract_info -f /path/to/file.pdf -p <page>

Process a PDF file and save the data as CSV for latter processing. The output csv can be passed to *run.py get_embeddings* script:

    python run.py extract_info -f /path/to/file.pdf -t csv -o /path/to/out.csv

Process all the PDF files in a directory and save a .txt file for each PDF file:

    python run.py extract_info -d /path/to/dir/ -o /path/to/out/

Process all the PDF files in a directory and save a .csv file for each PDF file. The output csv files can be passed to *run.py get_embeddings* script:

    python run.py extract_info -d /path/to/dir/ -t csv -o /path/to/out/

Keep cache of the file, usefull when the same file is going to be processed multiple times. *-k* option is compatible with all the other options:

    python run.py extract_info --loader mixed -f /path/to/file.pdf -t csv -o /path/to/out.csv -k
