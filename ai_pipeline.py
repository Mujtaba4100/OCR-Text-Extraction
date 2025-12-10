import os
import pytesseract
from pdf2image import convert_from_path
import re
import cv2
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import string

# OCR Config
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
poppler_path = r"C:\Library\Release-25.12.0-0\poppler-25.12.0\Library\bin"

# NLP Downloads
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words("english"))

# ------------- CLEAN TEXT -----------
def clean_text(text, preserve_numbers=False):
    text = text.lower()
    if not preserve_numbers:
        text = re.sub(r'\d+', ' ', text)  # Remove numbers only if not preserving
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    doc = nlp(" ".join(tokens))
    lemmatized = [token.lemma_ for token in doc]
    cleaned_tokens = [word for word in lemmatized if len(word) > 2]
    return " ".join(cleaned_tokens)

# ------------- OCR EXTRACT TEXT -----------
def extract_text(file_path):
    extracted_text = ""

    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path, poppler_path=poppler_path)
        for i, page in enumerate(pages):
            text = pytesseract.image_to_string(page)
            extracted_text += f"\n---- Page {i+1} ----\n{text}"

    elif file_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        img = cv2.imread(file_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray_img)
        extracted_text += text

    else:
        raise ValueError("Unsupported File")

    return extracted_text

# ------------- CNIC EXTRACTION -----------
import re

def extract_cnic(ocr_text):
    lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]
    
    data = {
        "name": None,
        "father_name": None,
        "gender": None,
        "country_of_stay": None,
        "identity_number": None,
        "date_of_birth": None,
        "date_of_issue": None,
        "date_of_expiry": None
    }

    # Regex patterns
    id_pattern = re.compile(r"\b\d{5}-?\d{7}-?\d\b")
    date_pattern = re.compile(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b")

    def clean(val):
        if not val: return None
        return val.replace(":", "").replace("}", "").replace("{", "").replace("|", "").replace("i", "").strip()

    for i, line in enumerate(lines):
        low = line.lower()

        # --- NAME ---
        if low.startswith("name"):
            data["name"] = clean(lines[i+1]) if i+1 < len(lines) else None

        # --- FATHER NAME ---
        elif low.startswith("father name"):
            data["father_name"] = clean(lines[i+1]) if i+1 < len(lines) else None

        # --- GENDER + COUNTRY (special case for your OCR) ---
        elif "gender" in low and "country" in low:
            if i+1 < len(lines):
                vals = lines[i+1].split()
                # Example: "M i Pakistan"
                if len(vals) >= 1: data["gender"] = clean(vals[0])
                if len(vals) >= 3: data["country_of_stay"] = clean(vals[-1])

        # --- ID + DOB on same line ---
        if id_pattern.search(line):
            data["identity_number"] = id_pattern.search(line).group()

            # Extract DOB separately
            dates = date_pattern.findall(line)
            if len(dates) >= 1:
                data["date_of_birth"] = dates[-1]  # last date = DOB

        # --- ISSUE + EXPIRY on same line ---
        elif "date of issue" in low or "date issue" in low:
            if i+1 < len(lines):
                next_line = lines[i+1]
                dates = date_pattern.findall(next_line)
                if len(dates) >= 1:
                    data["date_of_issue"] = dates[0].replace(" ", "")
                if len(dates) >= 2:
                    data["date_of_expiry"] = dates[1].replace(" ", "")

        # Fallback: collect missing dates
        if not data["date_of_issue"] or not data["date_of_expiry"]:
            dates = date_pattern.findall(line)
            for d in dates:
                if not data["date_of_issue"]:
                    data["date_of_issue"] = d
                elif not data["date_of_expiry"]:
                    data["date_of_expiry"] = d

    return data
# ------------- EMR EXTRACTION -----------
def extract_emr(raw_text):
    fields = {
        
        "Patient Name": None,
        "Age": None,
        "Gender": None,
        "Doctor": None,
        "Diagnosis": None,
        "Tests": None,
        "Medicines": None,
        "Date": None
    }

    text_lower = raw_text.lower()

    # --- Patient Name ---
    # Support variations: "Patient Name", "Name of Patient", "Pt Name"
    name_patterns = [
        r"patient name[:\- ]+([a-zA-Z ]+)",
        r"name of patient[:\- ]+([a-zA-Z ]+)",
        r"pt name[:\- ]+([a-zA-Z ]+)"
    ]
    for pat in name_patterns:
        match = re.search(pat, raw_text, re.I)
        if match:
            fields["Patient Name"] = match.group(1).strip()
            break

    # --- Gender ---
    # Handle "Male", "M", "Female", "F"
    if re.search(r"\bmale\b|\bm\b", text_lower):
        fields["Gender"] = "Male"
    elif re.search(r"\bfemale\b|\bf\b", text_lower):
        fields["Gender"] = "Female"

    # --- Age ---
    # Support "Age", "Patient Age"
    age_match = re.search(r"(?:age|patient age)[:\- ]*(\d+)", raw_text, re.I)
    if age_match:
        fields["Age"] = age_match.group(1)

    # --- Doctor / Physician ---
    doctor_patterns = [
        r"doctor[:\- ]+([a-zA-Z .]+)",
        r"physician[:\- ]+([a-zA-Z .]+)",
        r"attending doctor[:\- ]+([a-zA-Z .]+)"
    ]
    for pat in doctor_patterns:
        match = re.search(pat, raw_text, re.I)
        if match:
            fields["Doctor"] = match.group(1).strip()
            break

    # --- Diagnosis ---
    diag_patterns = [
        r"diagnosis[:\- ]+(.+)",
        r"dx[:\- ]+(.+)"
    ]
    for pat in diag_patterns:
        match = re.search(pat, raw_text, re.I)
        if match:
            fields["Diagnosis"] = match.group(1).strip()
            break

    # --- Date ---
    # Support formats like dd/mm/yyyy, dd-mm-yyyy, yyyy-mm-dd
    date_match = re.search(r"\b(\d{2}[./-]\d{2}[./-]\d{4}|\d{4}[./-]\d{2}[./-]\d{2})\b", raw_text)
    if date_match:
        fields["Date"] = date_match.group(0)

    # --- Medicines ---
    # Match patterns like "Paracetamol 500mg", "Amoxicillin 250mg", including optional mg/ml
    meds = re.findall(r"([A-Z][a-z]+(?: [A-Z][a-z]+)* \d+(?:mg|ml)?)", raw_text)
    if meds:
        fields["Medicines"] = ", ".join(meds)

    # --- Tests / Lab investigations ---
    # Support common abbreviations and keywords
    test_keywords = [
        "cbc", "lft", "rft", "xray", "ultrasound", "ecg", "mri", "ct scan", "blood sugar", "hb", "cholesterol"
    ]
    found_tests = set()
    for tk in test_keywords:
        if re.search(r"\b" + re.escape(tk) + r"\b", text_lower):
            found_tests.add(tk.upper())
    if found_tests:
        fields["Tests"] = ", ".join(sorted(found_tests))

    return fields

# ------------- ERP EXTRACTION -----------
def extract_erp(raw_text):
    fields = {
        "Document Type": "ERP Document",
        "Employee Name": None,
        "Employee ID": None,
        "Invoice No": None,
        "Salary Amount": None,
        "Department": None,
        "Date": None
    }

    # EMPLOYEE NAME
    emp_name = re.search(r"employee name[:\- ]+([a-zA-Z ]+)", raw_text, re.I)
    if emp_name: fields["Employee Name"] = emp_name.group(1).strip()

    # EMPLOYEE ID
    emp_id = re.search(r"employee id[:\- ]+([A-Za-z0-9\-]+)", raw_text)
    if emp_id: fields["Employee ID"] = emp_id.group(1)

    # INVOICE NO
    inv = re.search(r"(invoice no|invoice#)[:\- ]+([A-Za-z0-9\-]+)", raw_text, re.I)
    if inv: fields["Invoice No"] = inv.group(2)

    # SALARY AMOUNT
    salary = re.search(r"(salary|basic pay)[:\- ]+([\d,]+)", raw_text, re.I)
    if salary: fields["Salary Amount"] = salary.group(2)

    # DEPARTMENT
    dept = re.search(r"department[:\- ]+([a-zA-Z ]+)", raw_text, re.I)
    if dept: fields["Department"] = dept.group(1).strip()

    # DATE
    date = re.search(r"\d{2}[.\-/]\d{2}[.\-/]\d{4}", raw_text)
    if date: fields["Date"] = date.group(0)

    return fields

# ------------- PROPERTY EXTRACTION -----------
def extract_property(raw_text):
    fields = {
        "Document Type": "Property Document",
        "Owner Name": None,
        "Plot Number": None,
        "Khasra Number": None,
        "Area": None,
        "Registry Date": None,
    }

    # OWNER NAME
    owner = re.search(r"(owner name|title holder)[:\- ]+([a-zA-Z ]+)", raw_text, re.I)
    if owner: fields["Owner Name"] = owner.group(2).strip()

    # PLOT #
    plot = re.search(r"plot[:\- ]+([A-Za-z0-9\-]+)", raw_text, re.I)
    if plot: fields["Plot Number"] = plot.group(1)

    # KHASRA #
    khasra = re.search(r"khasra[:\- ]+([A-Za-z0-9\-]+)", raw_text, re.I)
    if khasra: fields["Khasra Number"] = khasra.group(1)

    # AREA
    area = re.search(r"area[:\- ]+([A-Za-z0-9 ]+)", raw_text, re.I)
    if area: fields["Area"] = area.group(1).strip()

    # REGISTRY DATE
    date = re.search(r"\d{2}[.\-/]\d{2}[.\-/]\d{4}", raw_text)
    if date: fields["Registry Date"] = date.group(0)

    return fields

# ------------- DOCUMENT TYPE DETECTION -----------
def detect_document_type(raw_text):
    text = raw_text.lower()

    if "identity card" in text or re.search(r"\d{5}-\d{7}-\d", raw_text):
        return "CNIC"

    if "patient" in text or "medical" in text or "doctor" in text or "diagnosis" in text:
        return "EMR"

    if "invoice" in text or "employee" in text or "salary" in text:
        return "ERP"

    if "plot" in text or "khasra" in text or "registry" in text:
        return "PROPERTY"

    return "UNKNOWN"

# ------------- FINAL PROCESS -----------
def process(file_path):
    raw_text = extract_text(file_path)
    doc_type = detect_document_type(raw_text)

    if doc_type == "CNIC":
        fields = extract_cnic(raw_text)
    elif doc_type == "EMR":
        fields = extract_emr(raw_text)
    elif doc_type == "ERP":
        fields = extract_erp(raw_text)
    elif doc_type == "PROPERTY":
        fields = extract_property(raw_text)
    else:
        fields = {"Error": "Unknown Document Type"}

    # Create uniform JSON structure
    json_output = {
        "DOCtype": doc_type,
        "rawtext": raw_text,
    }

    # Merge extracted fields into the standard format
    json_output.update(fields)

    return json_output
