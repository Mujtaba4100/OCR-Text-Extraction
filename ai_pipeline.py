import os
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path
import cv2
from PIL import Image
import google.generativeai as genai
import json
import re

load_dotenv()
api_key=os.getenv("api_key")
genai.configure(api_key=api_key)
print("API key fingerprint:", api_key)


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
poppler_path = r"C:\Library\Release-25.12.0-0\poppler-25.12.0\Library\bin"

def extract_text(file_path: str) -> str:
    """Extract text from PDF/image using OCR."""
    text = ""
    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path, poppler_path=poppler_path)
        for page in pages:
            text += pytesseract.image_to_string(page) + "\n"
    else:
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text += pytesseract.image_to_string(gray)
    return text


def detect_document_type(text: str) -> str:
    t = text.lower()
    if "identity card" in t or "cnic" in t or "nic" in t:
        return "CNIC"
    elif "patient" in t or "diagnosis" in t or "medical" in t:
        return "EMR"
    elif "invoice" in t or "salary" in t or "employee" in t:
        return "ERP"
    elif "plot" in t or "khasra" in t or "registry" in t:
        return "PROPERTY"
    else:
        return "UNKNOWN"

def generate_prompt(text: str, doc_type: str) -> str:
    """Build a strict JSON extraction prompt for Gemini."""
    if doc_type == "CNIC":
        fields = [
            "name", "father_name", "gender",
            "country_of_stay", "identity_number",
            "date_of_birth", "date_of_issue", "date_of_expiry"
        ]
    elif doc_type == "EMR":
        fields = ["patient_name", "age", "gender", "diagnosis", "prescription", "doctor_name", "date"]
    elif doc_type == "ERP":
        fields = ["invoice_number", "date", "supplier_name", "buyer_name", "amount", "items"]
    elif doc_type == "PROPERTY":
        fields = ["owner_name", "plot_number", "registry_number", "area", "location", "date_of_issue"]
    else:
        fields = []

    fields_list = ", ".join(f'"{f}"' for f in fields)
    return f"""
Extract ONLY the following fields in JSON format (null if missing):
{fields_list}

Text:
\"\"\"
{text}
\"\"\"
"""

def safe_json_loads(text: str) -> dict:
    """
    Try to parse JSON safely. If Gemini output is messy, extract {...}.
    """
    text = text.strip()
    if not text:
        return {"error": "Empty response from Gemini"}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return {"error": "Could not parse JSON from Gemini output"}
        return {"error": "No valid JSON found"}


def extract_fields_with_gemini(text: str, doc_type: str) -> dict:
    """Use Gemini to extract structured fields."""
    if doc_type == "UNKNOWN":
        return {"DOCtype": "UNKNOWN", "rawtext": text}

    prompt = generate_prompt(text, doc_type)

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        raw_output = response.text
        fields = safe_json_loads(raw_output)
    except Exception as e:
        fields = {"error": str(e)}

    fields["DOCtype"] = doc_type
    fields["rawtext"] = text
    return fields

def process(file_path: str) -> dict:
    """
    Complete pipeline:
    1. OCR extraction
    2. Document type detection
    3. Gemini-based field extraction
    """
    raw_text = extract_text(file_path)
    doc_type = detect_document_type(raw_text)
    return extract_fields_with_gemini(raw_text, doc_type)


