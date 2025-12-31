# AI Document Management System

A Python-based document processing system with OCR, AI extraction, and semantic search.

## Features

- **OCR Extraction** - Tesseract-based text extraction from PDFs and images
- **Document Detection** - Auto-detect CNIC, EMR, ERP, Property documents
- **AI Field Extraction** - Gemini-powered structured data extraction
- **Semantic Search** - FAISS + Sentence Transformers for natural language queries

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install flask pytesseract pdf2image opencv-python pillow
pip install google-generativeai python-dotenv
pip install sentence-transformers faiss-cpu
```

## Configuration

Create `.env` file:
```
api_key=YOUR_GEMINI_API_KEY
```

## Usage

### Start the App
```bash
python app.py
```
Open http://localhost:5000

### Semantic Search
```python
from embedding_pipeline import semantic_search

results = semantic_search("Find CNIC for Ali Khan", top_k=5)
```

## Project Structure

```
├── app.py                 # Flask web application
├── ai_pipeline.py         # OCR + Gemini extraction
├── embedding_pipeline.py  # FAISS semantic search
├── templates/
│   └── index.html
├── uploads/               # Uploaded documents
├── data/                  # Extracted JSON files
├── faiss.index            # Vector index
└── metadata.json          # Document metadata
```

## API

| Function | Description |
|----------|-------------|
| `process(file_path)` | Extract structured data from document |
| `add_document_to_index(doc_json)` | Index document for search |
| `semantic_search(query, top_k)` | Find similar documents |
