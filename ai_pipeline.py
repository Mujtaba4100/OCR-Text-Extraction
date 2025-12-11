# ai_pipeline.py
import os
import re
from typing import Dict, Any
from collections import defaultdict

# OCR libs
import pytesseract
from pdf2image import convert_from_path
import cv2
from PIL import Image

# NLP libs
import spacy
from transformers import pipeline

# ---- CONFIG: adjust these paths for your machine ----
# Windows example for Tesseract:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Poppler path for pdf2image (Windows example)
# POPPLER_PATH = r"C:\Library\Release-25.12.0-0\poppler-25.12.0\Library\bin"
POPPLER_PATH = os.getenv("POPPLER_PATH", None)

# Load spaCy once
spacy_nlp = spacy.load("en_core_web_sm")

# Lazy init for HF NER pipelines - only build when needed to save startup time
NER_PIPELINES = {}

def get_ner_pipeline(name: str):
    """Return a HuggingFace NER pipeline for a given domain (cached)."""
    if name in NER_PIPELINES:
        return NER_PIPELINES[name]
    # Map doc types to recommended NER models
    model_map = {
        "EMR": "dslim/bert-base-NER",
        "ERP": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "PROPERTY": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "CNIC": "dslim/bert-base-NER",
        "UNKNOWN": "dslim/bert-base-NER",
    }
    model_name = model_map.get(name, model_map["UNKNOWN"])
    ner = pipeline("ner", model=model_name, grouped_entities=True)
    NER_PIPELINES[name] = ner
    return ner

# ---------------- OCR / text extraction ----------------
def extract_text(file_path: str) -> str:
    """Extract text from PDF or image using pdf2image + pytesseract or pytesseract on image."""
    text = ""
    ext = file_path.lower().split(".")[-1]
    if ext == "pdf":
        # convert_from_path returns PIL images
        pages = convert_from_path(file_path, poppler_path=POPPLER_PATH) if POPPLER_PATH else convert_from_path(file_path)
        for i, page in enumerate(pages):
            page_rgb = page.convert("RGB")
            page_text = pytesseract.image_to_string(page_rgb)
            text += f"\n---- PAGE {i+1} ----\n{page_text}"
    elif ext in ("png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"):
        img = cv2.imread(file_path)
        if img is None:
            # fallback to PIL
            pil = Image.open(file_path)
            text = pytesseract.image_to_string(pil)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
    else:
        raise ValueError("Unsupported file type for OCR")
    return text

# ---------------- Document type detection ----------------
def detect_document_type(text: str) -> str:
    tl = text.lower()
    if any(k in tl for k in ("identity card", "cnic", "national identity card", "identity no", "nic")):
        return "CNIC"
    if any(k in tl for k in ("patient", "diagnosis", "prescription", "medical", "mrn", "medicine")):
        return "EMR"
    if any(k in tl for k in ("invoice", "salary", "pay", "amount", "employee", "salary slip", "payroll", "bill")):
        return "ERP"
    if any(k in tl for k in ("khasra", "plot", "registry", "registry", "property", "owner", "khewat")):
        return "PROPERTY"
    return "UNKNOWN"

# ---------------- NER extraction ----------------
def extract_entities(text: str, doc_type: str) -> Dict[str, list]:
    """Get entities using HF NER + spaCy backup."""
    entities = defaultdict(list)

    # HuggingFace NER
    try:
        ner = get_ner_pipeline(doc_type)
        hf_ents = ner(text)
        for ent in hf_ents:
            # grouped_entities: ent has 'entity_group' and 'word' or 'score'
            label = ent.get("entity_group", "MISC")
            word = ent.get("word") or ent.get("entity")
            # normalize spaces
            word = re.sub(r"\s+", " ", word).strip()
            entities[label].append(word)
    except Exception:
        # ignore HF errors, fallback to spaCy
        pass

    # spaCy as backup
    doc = spacy_nlp(text)
    for ent in doc.ents:
        entities[ent.label_].append(ent.text.strip())

    # deduplicate
    for k in list(entities.keys()):
        entities[k] = list(dict.fromkeys([v for v in entities[k] if v]))

    return dict(entities)

# ---------------- Rule-based extraction helpers ----------------
def find_cnic_number(text: str) -> str:
    # Common pattern: 5-7-1 digits
    m = re.search(r"\b\d{5}-\d{7}-\d\b", text)
    if m:
        return m.group(0)
    # alternative contiguous digits (13 digits)
    m2 = re.search(r"\b(\d{13})\b", text)
    if m2:
        s = m2.group(1)
        return f"{s[:5]}-{s[5:12]}-{s[12:]}"
    return ""

def find_dates(text: str) -> list:
    date_patterns = [
        r"\b\d{2}[./-]\d{2}[./-]\d{4}\b",
        r"\b\d{2}[./-]\d{2}[./-]\d{2}\b",
        r"\b\d{2}[ ]+[A-Za-z]{3,9}[ ]+\d{4}\b",
        r"\b[A-Za-z]{3,9}[ ]+\d{1,2},[ ]+\d{4}\b"
    ]
    dates = []
    for p in date_patterns:
        for m in re.findall(p, text):
            if m not in dates:
                dates.append(m)
    return dates

def find_amounts(text: str) -> list:
    # common currency/amount patterns
    patterns = [
        r"\bRs\.?\s?[\d,]+(?:\.\d+)?\b",
        r"\bPKR\s?[\d,]+(?:\.\d+)?\b",
        r"\b[0-9]{1,3}(?:,[0-9]{3})+(?:\.\d+)?\b",
        r"\b\d+(?:\.\d{1,2})?\s?(?:USD|PKR|Rs|Rs\.)?\b"
    ]
    amounts = []
    for p in patterns:
        for m in re.findall(p, text, flags=re.IGNORECASE):
            m = m.strip()
            if m not in amounts:
                amounts.append(m)
    return amounts

def find_invoice_numbers(text: str) -> list:
    patterns = [r"\bINV[-\s]?\d{2,6}\b", r"\bInvoice\s*#?:?\s*\d{1,8}\b", r"\bBill\s*#?:?\s*\d{1,8}\b"]
    invs = []
    for p in patterns:
        for m in re.findall(p, text, flags=re.IGNORECASE):
            if m not in invs:
                invs.append(m.strip())
    return invs

def find_names_by_keyword(text: str, keyword_list: list) -> str:
    # search for lines containing keywords then extract name-like substring after colon or on same line
    lines = text.splitlines()
    for i, line in enumerate(lines):
        for kw in keyword_list:
            if kw.lower() in line.lower():
                # after colon
                if ":" in line:
                    val = line.split(":", 1)[1].strip()
                    if val:
                        return re.sub(r"[^A-Za-z\s'-]", "", val).strip()
                # maybe next line is the value
                if i + 1 < len(lines):
                    val = lines[i + 1].strip()
                    if val:
                        return re.sub(r"[^A-Za-z\s'-]", "", val).strip()
    return ""

# ---------------- Document-specific rule extractors ----------------
def extract_cnic_fields(text: str, entities: Dict[str, list]) -> Dict[str, Any]:
    data = {}
    data["cnic_number"] = find_cnic_number(text)
    # Name extraction: prefer keyword "Name" then PERSON entities
    name = find_names_by_keyword(text, ["name", "naam"])
    if not name:
        name_candidates = entities.get("PERSON") or entities.get("PER") or []
        if name_candidates:
            name = name_candidates[0]
    data["name"] = name

    father = find_names_by_keyword(text, ["father", "father name", "s/o", "son of", "walid"])
    if not father:
        # try second PERSON
        person_list = entities.get("PERSON") or entities.get("PER") or []
        if len(person_list) > 1:
            father = person_list[1]
    data["father_name"] = father

    dates = find_dates(text)
    # heuristics: DOB usually earlier in text and by keyword
    dob = ""
    issue = ""
    expiry = ""
    for d in dates:
        if re.search(r"date of birth|dob", d, flags=re.IGNORECASE) or re.search(r"date of birth|dob", text, flags=re.IGNORECASE):
            dob = d
    # fallback: take first date as DOB if looks plausible
    if not dob and dates:
        dob = dates[0]
    # issue/expiry detection by keywords
    issue_match = re.search(r"date of issue[:\s]*([0-9./\-\sA-Za-z,]+)", text, flags=re.IGNORECASE)
    expiry_match = re.search(r"date of (?:expi|expir)[\w\s:]*([0-9./\-\sA-Za-z,]+)", text, flags=re.IGNORECASE)
    if issue_match:
        issue = issue_match.group(1).strip().split("\n")[0]
    if expiry_match:
        expiry = expiry_match.group(1).strip().split("\n")[0]

    data["date_of_birth"] = dob
    data["issue_date"] = issue
    data["expiry_date"] = expiry
    return data

def extract_erp_fields(text: str, entities: Dict[str, list]) -> Dict[str, Any]:
    data = {}
    # names
    name = find_names_by_keyword(text, ["employee", "name", "employee name", "staff"])
    if not name:
        person_candidates = entities.get("PERSON") or entities.get("PER") or []
        name = person_candidates[0] if person_candidates else ""
    data["names"] = [name] if name else []

    # invoice / ids
    invs = find_invoice_numbers(text)
    data["invoice_numbers"] = invs

    # amounts
    amounts = find_amounts(text)
    data["amounts"] = amounts

    # dates
    data["dates"] = find_dates(text)

    # employee id detection
    emp_ids = re.findall(r"\bEMP[-\s]?\d{1,6}\b", text, flags=re.IGNORECASE)
    data["employee_ids"] = list(dict.fromkeys(emp_ids))

    # try to detect salary/total lines with keywords
    totals = []
    for line in text.splitlines():
        if any(k in line.lower() for k in ("total", "net pay", "gross pay", "net salary", "salary", "amount payable", "amount:")):
            m = re.search(r"([0-9\.,]+(?:\s?(?:PKR|Rs|USD)?)?)", line)
            if m:
                val = m.group(1).strip()
                if val not in totals:
                    totals.append(val)
    if totals:
        data["totals"] = totals
    return data

def extract_emr_fields(text: str, entities: Dict[str, list]) -> Dict[str, Any]:
    data = {}
    # patient name
    name = find_names_by_keyword(text, ["patient", "patient name", "name"])
    if not name:
        person_candidates = entities.get("PERSON") or entities.get("PER") or []
        name = person_candidates[0] if person_candidates else ""
    data["patient_name"] = name

    # doctor
    doctor = find_names_by_keyword(text, ["doctor", "dr.", "physician", "consultant"])
    if not doctor:
        # try PERSON with "Dr" nearby
        m = re.search(r"(Dr\.?\s*[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)", text)
        if m:
            doctor = m.group(1)
    data["doctor"] = doctor

    # diagnosis (look for blocks after 'Diagnosis' or 'Impression')
    diag = ""
    m = re.search(r"(Diagnosis|Impression|Assessment)[:\s]*([\s\S]{0,300}?)\n\n", text, flags=re.IGNORECASE)
    if m:
        diag = m.group(2).strip()
    else:
        m2 = re.search(r"(Diagnosis|Impression|Assessment)[:\s]*([^\n]+)", text, flags=re.IGNORECASE)
        if m2:
            diag = m2.group(2).strip()
    data["diagnosis"] = diag

    # medicines: find "Rx" or "Prescription" sections
    meds = []
    med_block = ""
    m3 = re.search(r"(Prescription|Medications|Rx)[:\s]*([\s\S]{0,500})$", text, flags=re.IGNORECASE)
    if m3:
        med_block = m3.group(2)
    if med_block:
        for line in med_block.splitlines():
            if line.strip():
                meds.append(line.strip())
    data["medicines"] = meds
    data["dates"] = find_dates(text)
    return data

def extract_property_fields(text: str, entities: Dict[str, list]) -> Dict[str, Any]:
    data = {}
    # owner
    owner = find_names_by_keyword(text, ["owner", "proprietor", "possessor"])
    if not owner:
        person_candidates = entities.get("PERSON") or entities.get("PER") or []
        owner = person_candidates[0] if person_candidates else ""
    data["owner"] = owner

    # plot / khasra
    plot = ""
    m = re.search(r"(khasra|plot|khewat|khata)\s*[:#]?\s*([A-Za-z0-9\-\/]+)", text, flags=re.IGNORECASE)
    if m:
        plot = m.group(2).strip()
    data["plot_or_khasra"] = plot

    # area / size
    area = ""
    m2 = re.search(r"\b(area|size)\b[:\s]*([0-9\.,]+\s*(?:acre|acres|sq\.? ft|sqm|sqft|marla)?)", text, flags=re.IGNORECASE)
    if m2:
        area = m2.group(2).strip()
    data["area"] = area

    # registry / deed numbers
    reg = re.findall(r"\bREG[-\s]?\d{1,6}\b", text, flags=re.IGNORECASE)
    data["registry_numbers"] = reg
    return data

# ---------------- Final processing ----------------
def rule_based_extract(text: str, doc_type: str, entities: Dict[str, list]) -> Dict[str, Any]:
    """Return advanced JSON with 'fields', 'entities_raw', and 'text'"""
    result = {
        "doc_type": doc_type,
        "fields": {},
        "entities_raw": entities,
        "text": text
    }
    if doc_type == "CNIC":
        result["fields"] = extract_cnic_fields(text, entities)
    elif doc_type == "ERP":
        result["fields"] = extract_erp_fields(text, entities)
    elif doc_type == "EMR":
        result["fields"] = extract_emr_fields(text, entities)
    elif doc_type == "PROPERTY":
        result["fields"] = extract_property_fields(text, entities)
    else:
        # Generic extraction for unknowns: numbers, dates, persons, amounts
        result["fields"] = {
            "names": entities.get("PERSON") or entities.get("PER") or [],
            "dates": find_dates(text),
            "amounts": find_amounts(text),
            "invoices": find_invoice_numbers(text)
        }
    return result

def process(file_path: str) -> Dict[str, Any]:
    """Full pipeline: OCR -> doc detection -> NER -> rule-based extraction"""
    raw_text = extract_text(file_path)
    doc_type = detect_document_type(raw_text)
    entities = extract_entities(raw_text, doc_type)
    structured = rule_based_extract(raw_text, doc_type, entities)
    # Add a simple confidence heuristic (presence of fields)
    structured["meta"] = {
        "num_entities": sum(len(v) for v in entities.values()),
        "has_text": bool(raw_text.strip()),
        "cnic_detected": bool(structured["fields"].get("cnic_number"))
    }
    return structured

# For testing quick CLI
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) > 1:
        out = process(sys.argv[1])
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print("ai_pipeline.py - call process(path_to_file)")
