# Bajaj OCR API

Simple HTTP service to pull line items and totals from invoices (images or PDFs). Built with FastAPI and Tesseract OCR.

## What it does
- Extracts line items with amounts.
- Attempts to read sub‑total, tax and final total from the document text.
- Works with multi‑page PDFs (aggregates page results into a single response).

## Setup

1. Create a virtual environment and install packages
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. System prerequisites
   - Tesseract OCR (e.g., macOS: `brew install tesseract`)
   - Poppler for PDF to image (e.g., macOS: `brew install poppler`)

## Run the server
```bash
uvicorn app.main:app --reload --port 8000
```
Open http://127.0.0.1:8000 to check it’s up. Swagger UI is at http://127.0.0.1:8000/docs

## Usage

Endpoint: `POST /extract`

Example (image):
```bash
curl -s -X POST http://127.0.0.1:8000/extract \
  -H "accept: application/json" \
  -F "file=@Data/sample_1.png"
```

Example (PDF):
```bash
curl -s -X POST http://127.0.0.1:8000/extract \
  -H "accept: application/json" \
  -F "file=@Data/sample_2.pdf"
```

Save response to a file:
```bash
curl -s -X POST http://127.0.0.1:8000/extract \
  -F "file=@Data/sample_2.pdf" -o output.json
```

### Response shape (current)
```json
{
  "filename": "sample_2.pdf",
  "data": {
    "line_items": [
      {"description": "Service A", "quantity": null, "rate": null, "amount": 1200.00}
    ],
    "sub_total": 1200.00,
    "tax": 216.00,
    "grand_total": 1416.00
  },
  "success": true,
  "error": null
}
```

Notes:
- For PDFs, pages are converted to images and processed one by one.
- Line items from all pages are merged; the largest detected grand total is used as final total. If none is found, it falls back to summing item amounts.

## Project Structure
- `app/` – source
  - `core/` – OCR, preprocessing and extraction
  - `models/` – response/request schemas
  - `main.py` – API entry point
- `Data/` – sample files

## Matching external JSON specs
If you need to match a specific competition schema, share the JSON template and the API will be adjusted to return exactly those fields.