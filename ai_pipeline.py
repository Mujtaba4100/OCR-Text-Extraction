import os
import json
import re
import requests
import cv2
import pytesseract
from dotenv import load_dotenv
from pdf2image import convert_from_path

# =========================
# ENV & CONFIG
# =========================
load_dotenv()
API_KEY = os.getenv("api_key")
API_MODEL = os.getenv("api_model", "llama-3.3-70b-versatile")

if not API_KEY:
    raise RuntimeError("API key not found. Please set 'api_key' in .env")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Library\Release-25.12.0-0\poppler-25.12.0\Library\bin"

# =========================
# GROQ API CALL
# =========================
def call_groq_chat(prompt: str, model: str = None) -> str:
    """Call Groq OpenAI-compatible Chat Completions API."""
    final_model = model or API_MODEL

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": final_model,
        "messages": [
            {"role": "system", "content": "You are a strict JSON extraction engine."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 2048
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f"Groq API Error {resp.status_code}: {resp.text[:1000]}")

    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

# =========================
# OCR EXTRACTION
# =========================
def extract_text(file_path: str) -> str:
    """Extract text from PDF or image using OCR."""
    text = ""

    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path, poppler_path=POPPLER_PATH)
        for page in pages:
            text += pytesseract.image_to_string(page) + "\n"
    else:
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)

    return text.strip()

# =========================
# DOCUMENT TYPE DETECTION
# =========================
def detect_document_type(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["identity card", "cnic", "nic"]):
        return "CNIC"
    if any(k in t for k in ["patient", "diagnosis", "medical"]):
        return "EMR"
    if any(k in t for k in ["invoice", "salary", "employee"]):
        return "ERP"
    if any(k in t for k in ["plot", "khasra", "registry"]):
        return "PROPERTY"
    return "UNKNOWN"

# =========================
# PROMPT GENERATION
# =========================
def generate_prompt(text: str, doc_type: str) -> str:
    fields_map = {
        "CNIC": [
            "name", "father_name", "gender", "country_of_stay",
            "identity_number", "date_of_birth",
            "date_of_issue", "date_of_expiry"
        ],
        "EMR": [
            "patient_name", "age", "gender",
            "diagnosis", "prescription",
            "doctor_name", "date"
        ],
        "ERP": [
            "invoice_number", "date", "supplier_name",
            "buyer_name", "amount", "items"
        ],
        "PROPERTY": [
            "owner_name", "plot_number",
            "registry_number", "area",
            "location", "date_of_issue"
        ]
    }

    fields = fields_map.get(doc_type, [])
    fields_list = ", ".join(f'"{f}"' for f in fields)

    return f"""
Extract ONLY the following fields in valid JSON.
Return null for missing values.
DO NOT add explanations.

Fields:
{fields_list}

Text:
\"\"\"
{text}
\"\"\"
"""

# =========================
# SAFE JSON PARSER
# =========================
def safe_json_loads(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"error": "Invalid JSON response", "raw_output": text}

# =========================
# FIELD EXTRACTION (GROQ)
# =========================
def extract_fields_with_groq(text: str, doc_type: str) -> dict:
    if doc_type == "UNKNOWN":
        return {"DOCtype": "UNKNOWN", "rawtext": text}

    prompt = generate_prompt(text, doc_type)
    raw_output = call_groq_chat(prompt)
    fields = safe_json_loads(raw_output)

    fields["DOCtype"] = doc_type
    fields["rawtext"] = text
    return fields

# =========================
# PIPELINE
# =========================
def process(file_path: str) -> dict:
    raw_text = extract_text(file_path)
    doc_type = detect_document_type(raw_text)
    return extract_fields_with_groq(raw_text, doc_type)
