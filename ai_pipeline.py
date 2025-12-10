import os
import torch
import pytesseract
from pdf2image import convert_from_path
import cv2
from PIL import Image
from typing import Dict
from collections import defaultdict
import spacy
from transformers import pipeline

# -------- OCR CONFIG ----------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
poppler_path = r"C:\Library\Release-25.12.0-0\poppler-25.12.0\Library\bin"

# -------- LOAD NLP MODELS ----------
# General English spaCy model
spacy_nlp = spacy.load("en_core_web_sm")

# HuggingFace Transformers NER pipelines for domain-specific tasks
# Future-proof: swap with med7, FinBERT, BERT-multilingual as needed
ner_models = {
    "EMR": pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True),
    "ERP": pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True),
    "PROPERTY": pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True),
    "CNIC": pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True),  # generic PERSON/DATE
}

# -------- OCR FUNCTION ----------
def extract_text(file_path: str) -> str:
    """Extract text from PDF/image using OCR"""
    text = ""
    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path, poppler_path=poppler_path)
        for i, page in enumerate(pages):
            page_text = pytesseract.image_to_string(page)
            text += f"\n---- Page {i+1} ----\n{page_text}"
    elif file_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        img = cv2.imread(file_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray_img)
    else:
        raise ValueError("Unsupported file type")
    return text

# -------- DOCUMENT TYPE DETECTION ----------
def detect_document_type(text: str) -> str:
    """Detect document type using keywords"""
    text_lower = text.lower()
    if "identity card" in text_lower or "cnic" in text_lower or "nic" in text_lower:
        return "CNIC"
    elif "patient" in text_lower or "diagnosis" in text_lower or "medical" in text_lower:
        return "EMR"
    elif "invoice" in text_lower or "salary" in text_lower or "employee" in text_lower:
        return "ERP"
    elif "plot" in text_lower or "khasra" in text_lower or "registry" in text_lower:
        return "PROPERTY"
    else:
        return "UNKNOWN"

# -------- ENTITY EXTRACTION ----------
def extract_entities(text: str, doc_type: str) -> Dict[str, list]:
    """
    Automatic NER extraction using domain-specific HF Transformers + spaCy backup
    Returns dict of entities grouped by type
    """
    entities = defaultdict(list)

    if doc_type in ner_models:
        ner = ner_models[doc_type]
        hf_entities = ner(text)
        for ent in hf_entities:
            label = ent['entity_group']
            entities[label].append(ent['word'])

    # Backup with spaCy
    doc = spacy_nlp(text)
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)

    # Deduplicate
    entities = {k: list(set(v)) for k, v in entities.items()}
    return entities

# -------- FINAL PROCESS FUNCTION ----------
def process(file_path: str) -> Dict:
    """
    Full pipeline:
    1. OCR extraction
    2. Document type detection
    3. Domain-aware NER
    """
    raw_text = extract_text(file_path)
    doc_type = detect_document_type(raw_text)
    entities = extract_entities(raw_text, doc_type)

    output = {
        "DOCtype": doc_type,
        "rawtext": raw_text,
        "entities": entities
    }
    return output
