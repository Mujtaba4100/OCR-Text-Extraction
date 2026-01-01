import os
import time
import logging
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path
import cv2
from PIL import Image
import google.generativeai as genai
import json
import re

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

genai.configure(api_key=os.getenv("api_key"))

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


def extract_fields_with_gemini(text: str, doc_type: str, max_retries: int = 3) -> dict:
    """Use Gemini to extract structured fields with retry logic."""
    if doc_type == "UNKNOWN":
        return {"DOCtype": "UNKNOWN", "rawtext": text}

    prompt = generate_prompt(text, doc_type)
    
    # Try multiple models in order of preference (using correct model names)
    models_to_try = [
        "gemini-1.5-pro",  # Stable model with good limits
        "gemini-1.5-flash-latest",  # Latest flash version
        "gemini-2.0-flash-exp",  # Experimental
        "gemini-exp-1206"  # Alternative experimental
    ]
    
    last_error = None
    
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                logging.info(f"[Gemini] Attempt {attempt + 1}/{max_retries} using model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                raw_output = response.text
                fields = safe_json_loads(raw_output)
                logging.info(f"[Gemini] Success with model: {model_name}")
                
                fields["DOCtype"] = doc_type
                fields["rawtext"] = text
                return fields
                
            except Exception as e:
                last_error = str(e)
                error_str = str(e).lower()
                
                # Check if it's a quota/rate limit error
                if "429" in error_str or "quota" in error_str or "rate limit" in error_str:
                    # Extract wait time if available
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    
                    # Try to extract suggested retry delay from error message
                    import re
                    retry_match = re.search(r'retry in (\d+\.?\d*)', error_str)
                    if retry_match:
                        wait_time = float(retry_match.group(1))
                    
                    if attempt < max_retries - 1:
                        logging.warning(f"[Gemini] Rate limit hit with {model_name}. Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"[Gemini] Max retries reached for {model_name}. Trying next model...")
                        break  # Try next model
                else:
                    # Non-quota error, log and break
                    logging.error(f"[Gemini] Error with {model_name}: {e}")
                    break  # Try next model
    
    # All models and retries failed
    fields = {
        "error": f"All Gemini models exhausted. Last error: {last_error}",
        "suggestion": "Please wait a few minutes or create a new Google Cloud project with a fresh API key."
    }
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
