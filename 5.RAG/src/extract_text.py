import fitz  # PyMuPDF
import os

BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "raw_pdfs"))
TEXT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "processed_texts"))
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

def pdf_to_text(pdf_path, txt_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

def process_all_pdfs():
    for pdf in os.listdir(RAW_DIR):
        if pdf.endswith(".pdf"):
            txt_name = pdf.replace(".pdf", ".txt")
            pdf_to_text(os.path.join(RAW_DIR, pdf),
                        os.path.join(TEXT_DIR, txt_name))

if __name__ == "__main__":
    if not os.path.exists(RAW_DIR) or not os.listdir(RAW_DIR):
        print(f"⚠️  No PDFs found in {RAW_DIR}")
        print("Please add PDF files to data/raw_pdfs/")
        exit(1)
    process_all_pdfs()
    print("✅ All PDFs converted to text.")
