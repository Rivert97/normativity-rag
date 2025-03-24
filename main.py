from Loaders import PdfMixedLoader
import os
from dotenv import load_dotenv
import glob

load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")
#PDF_PATH = os.path.join(STORAGE_DIR, "reglamento-academico-de-la-universidad-de-guanajuato.pdf")
#PDF_PATH = os.path.join(STORAGE_DIR, "ley-organica-de-la-universidad-de-guanajuato.pdf")
for file in glob.glob(f'{STORAGE_DIR}/*.pdf')[2:]:
    print(f"Procesando: {file}")
    pdf_loader = PdfMixedLoader(file)
    text = pdf_loader.get_text()
    #text = pdf_loader.get_page_text(17)

    basename = ''.join(os.path.basename(file).split('.')[:-1])
    with open(f"outs/{basename}.txt", 'w') as f:
        f.write(text)

#pdf_loader = PdfMixedLoader(PDF_PATH)
#text = pdf_loader.get_page_text(7)
#text = pdf_loader.get_text()
#print(text)