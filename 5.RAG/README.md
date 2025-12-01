# CyberSec RAG Analyzer

CyberSec RAG Analyzer is a Retrieval-Augmented Generation (RAG) application for querying cybersecurity documents. It combines semantic search (FAISS + Sentence Transformers) with a generative model (BART) to produce grounded answers with source attribution. The app includes a web UI built with Streamlit and supports uploading PDFs for instant analysis.

---

## Features

- **PDF upload in browser**: Upload one or more PDFs and analyze them immediately.
- **Dual-source search**: Query both pre-indexed documents and newly uploaded PDFs.
- **Semantic retrieval**: FAISS vector search over SentenceTransformer embeddings.
- **Grounded answers**: BART generates answers using retrieved context, with transparent source snippets.
- **Local processing**: No external API required; models are loaded locally.
- **Robust error handling**: Handles missing indices, large files, scanned PDFs, and model-loading edge cases.

---

## Architecture Overview

- **UI**: `src/app.py` (Streamlit)
- **Text extraction**: `src/extract_text.py` (PyMuPDF)
- **Embedding/index build**: `src/create_embeddings.py` (Sentence Transformers + FAISS)
- **Retrieval**: `src/retrieve_context.py` (loads FAISS index and searches)
- **Generation**: `src/generate_answer.py` (BART for conditional generation)

Data flow:
1. Text is extracted from PDFs and chunked into passages.
2. Passages are embedded and stored in a FAISS index.
3. User query is embedded and used to retrieve top-k passages.
4. Passages are concatenated as context for the BART model to answer.

---

## Project Structure

```
RAG-Cyber-analyzer/
├── data/
│   ├── raw_pdfs/             # your PDFs (input for offline pipeline)
│   └── processed_texts/      # extracted .txt files
├── embeddings/
│   └── faiss_index/          # FAISS index + texts/metadata numpy files
├── src/
│   ├── app.py                # Streamlit UI (supports PDF upload + Q&A)
│   ├── extract_text.py       # PDF → text converter (offline pipeline)
│   ├── create_embeddings.py  # build FAISS from processed_texts (offline)
│   ├── retrieve_context.py   # retrieval over FAISS (pre-indexed)
│   └── generate_answer.py    # answer generation with BART
├── requirements.txt
└── README.md
```

---

## Local Setup

- **Python**: 3.10+ recommended (tested also with 3.13)
- **OS**: macOS/Linux/Windows

### 1) Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```
If you do not use the requirements file, install individually:
```bash
pip install streamlit transformers sentence-transformers torch numpy faiss-cpu PyMuPDF
```

> Note for Apple Silicon (arm64): `faiss-cpu` wheels are available. The `torch` CPU wheel is used by default.

---

## Running the App

You can run the app either with pre-indexed documents or with only uploaded PDFs.

### Option A: Use uploaded PDFs only
```bash
streamlit run src/app.py
```
- In the sidebar, upload one or more PDF files.
- Click "Process Uploaded PDFs".
- Enter your question and click "Analyze Query".

### Option B: Pre-index documents (offline pipeline)
1. Place PDFs in `data/raw_pdfs/`.
2. Extract text to `data/processed_texts/`:
```bash
python src/extract_text.py
```
3. Build embeddings and FAISS index in `embeddings/faiss_index/`:
```bash
python src/create_embeddings.py
```
4. Start the app:
```bash
streamlit run src/app.py
```
5. In the main view, leave "Search existing documents" checked. You can also upload additional PDFs and enable "Search uploaded PDFs".

---

## Usage Notes

- **Search sources**: Use the checkboxes to select existing pre-indexed documents, uploaded PDFs, or both.
- **Context sources**: Retrieved passages are shown with their source filename and type (Existing Document or Uploaded PDF).
- **Answer visibility**: The UI styles ensure legible text on light backgrounds.

---

## Configuration

Relevant parameters are defined in code:
- `src/retrieve_context.py`
  - `INDEX_DIR`: location of the FAISS files (`embeddings/faiss_index/`)
  - `get_relevant_chunks(query, top_k=3)` selects top-k passages
- `src/app.py`
  - Chunk size for uploaded PDFs (`chunk_text` function)
  - File size limit (default 50 MB) for uploads
  - UI controls: checkboxes for source selection

---

## Deployment (Streamlit Cloud)

1. Ensure `requirements.txt` exists in repo root and includes:
```text
streamlit>=1.28.0
transformers>=4.35.0
sentence-transformers>=2.2.0
torch>=2.0.0
numpy>=1.24.0
faiss-cpu>=1.7.4
PyMuPDF>=1.23.0
```
2. Push to your Git repository.
3. Create a new Streamlit Cloud app pointed at this repo, and set the entry point to `src/app.py`.

> If you see "ModuleNotFoundError: faiss": verify `requirements.txt` is present and includes `faiss-cpu`.

---

## Troubleshooting

- **Meta tensor / device errors (PyTorch)**
  - The app pins models to CPU and uses robust loaders. If you still see errors, clear caches and restart the app.
- **FAISS index not found**
  - The UI shows a warning when the index is missing. Build the index via `src/create_embeddings.py` or use uploaded PDFs.
- **Scanned/image-based PDFs**
  - Text extraction may produce empty output. Use OCR preprocessing or upload text-based PDFs.
- **Large files / memory**
  - Upload limit is set to 50 MB per file in the app. Very large PDFs may require offline processing.
- **Slow first run**
  - Model downloads occur on first run. Subsequent runs should be faster due to caching.

---

## Security and Privacy

- All document processing and inference occur locally in your environment or in your Streamlit Cloud instance.
- No external APIs are called for embedding or generation.
- Treat uploaded files as sensitive. Do not commit private documents to the repository.

---

## License

This project is provided under the LICENSE file in the repository. Review and comply with the terms before use in production.
