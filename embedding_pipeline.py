"""
Embedding Pipeline Module for AI Document Management System
============================================================
This module provides embedding generation and semantic search capabilities
using Sentence Transformers and FAISS.

Features:
- Generate embeddings from structured document JSON
- Store vectors with metadata in FAISS index
- Perform semantic search across all indexed documents
- Persistent storage (faiss.index + metadata.json)

Usage:
    from embedding_pipeline import add_document_to_index, semantic_search
    
    # Add a document to the index
    add_document_to_index(doc_json)
    
    # Search for similar documents
    results = semantic_search("Find CNIC for Ali Khan", top_k=5)

Author: AI Document Management System
Version: 1.0.0
"""

import os
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# File paths for persistent storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss.index")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.json")

# =============================================================================
# GLOBAL INSTANCES (Lazy Loading)
# =============================================================================

_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.IndexFlatIP] = None
_metadata: List[Dict[str, Any]] = []


def _get_model() -> SentenceTransformer:
    """
    Lazy load the SentenceTransformer model.
    Returns the cached model instance if already loaded.
    """
    global _model
    if _model is None:
        print(f"[Embedding Pipeline] Loading model: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"[Embedding Pipeline] Model loaded successfully")
    return _model


def _get_index() -> faiss.IndexFlatIP:
    """
    Get or create the FAISS index.
    Loads from disk if exists, otherwise creates a new index.
    Uses Inner Product (IP) for cosine similarity with normalized vectors.
    """
    global _index, _metadata
    
    if _index is None:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
            # Load existing index and metadata
            print(f"[Embedding Pipeline] Loading existing FAISS index from disk")
            _index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                _metadata = json.load(f)
            print(f"[Embedding Pipeline] Loaded {len(_metadata)} documents from index")
        else:
            # Create new index
            print(f"[Embedding Pipeline] Creating new FAISS index")
            _index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            _metadata = []
    
    return _index


def _save_index() -> None:
    """
    Persist the FAISS index and metadata to disk.
    """
    global _index, _metadata
    
    if _index is not None:
        faiss.write_index(_index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(_metadata, f, indent=2, ensure_ascii=False)
        print(f"[Embedding Pipeline] Index saved to disk ({len(_metadata)} documents)")


# =============================================================================
# TEXT EXTRACTION STRATEGIES BY DOCUMENT TYPE
# =============================================================================

def _extract_text_cnic(data: Dict[str, Any]) -> str:
    """
    Extract searchable text from CNIC document.
    Fields: name, father_name, identity_number, date_of_birth, date_of_issue, date_of_expiry
    """
    fields = []
    
    if data.get("name"):
        fields.append(f"Name: {data['name']}")
    if data.get("father_name"):
        fields.append(f"Father Name: {data['father_name']}")
    if data.get("identity_number"):
        fields.append(f"CNIC Number: {data['identity_number']}")
    if data.get("date_of_birth"):
        fields.append(f"Date of Birth: {data['date_of_birth']}")
    if data.get("date_of_issue"):
        fields.append(f"Issue Date: {data['date_of_issue']}")
    if data.get("date_of_expiry"):
        fields.append(f"Expiry Date: {data['date_of_expiry']}")
    if data.get("gender"):
        fields.append(f"Gender: {data['gender']}")
    if data.get("address"):
        fields.append(f"Address: {data['address']}")
    
    return " | ".join(fields) if fields else "CNIC Document"


def _extract_text_emr(data: Dict[str, Any]) -> str:
    """
    Extract searchable text from EMR (Electronic Medical Record) document.
    Fields: diagnosis, medicines, doctor_name, patient_name, etc.
    """
    fields = []
    
    if data.get("patient_name"):
        fields.append(f"Patient: {data['patient_name']}")
    if data.get("doctor_name"):
        fields.append(f"Doctor: {data['doctor_name']}")
    if data.get("diagnosis"):
        diagnosis = data['diagnosis']
        if isinstance(diagnosis, list):
            diagnosis = ", ".join(diagnosis)
        fields.append(f"Diagnosis: {diagnosis}")
    if data.get("medicines"):
        medicines = data['medicines']
        if isinstance(medicines, list):
            medicines = ", ".join([str(m) for m in medicines])
        fields.append(f"Medicines: {medicines}")
    if data.get("symptoms"):
        symptoms = data['symptoms']
        if isinstance(symptoms, list):
            symptoms = ", ".join(symptoms)
        fields.append(f"Symptoms: {symptoms}")
    if data.get("hospital_name"):
        fields.append(f"Hospital: {data['hospital_name']}")
    if data.get("date"):
        fields.append(f"Date: {data['date']}")
    
    return " | ".join(fields) if fields else "Medical Record"


def _extract_text_erp(data: Dict[str, Any]) -> str:
    """
    Extract searchable text from ERP/Invoice document.
    Fields: invoice_number, items, amount, vendor, etc.
    """
    fields = []
    
    if data.get("invoice_number"):
        fields.append(f"Invoice: {data['invoice_number']}")
    if data.get("vendor_name") or data.get("company_name"):
        vendor = data.get("vendor_name") or data.get("company_name")
        fields.append(f"Vendor: {vendor}")
    if data.get("customer_name"):
        fields.append(f"Customer: {data['customer_name']}")
    if data.get("items"):
        items = data['items']
        if isinstance(items, list):
            item_names = [str(item.get("name", item) if isinstance(item, dict) else item) for item in items]
            fields.append(f"Items: {', '.join(item_names)}")
    if data.get("total_amount") or data.get("amount"):
        amount = data.get("total_amount") or data.get("amount")
        fields.append(f"Amount: {amount}")
    if data.get("date") or data.get("invoice_date"):
        date = data.get("date") or data.get("invoice_date")
        fields.append(f"Date: {date}")
    
    return " | ".join(fields) if fields else "Invoice/ERP Document"


def _extract_text_property(data: Dict[str, Any]) -> str:
    """
    Extract searchable text from Property document.
    Fields: owner_name, location, area, property_type, etc.
    """
    fields = []
    
    if data.get("owner_name"):
        fields.append(f"Owner: {data['owner_name']}")
    if data.get("property_type"):
        fields.append(f"Type: {data['property_type']}")
    if data.get("location") or data.get("address"):
        location = data.get("location") or data.get("address")
        fields.append(f"Location: {location}")
    if data.get("area"):
        fields.append(f"Area: {data['area']}")
    if data.get("plot_number"):
        fields.append(f"Plot: {data['plot_number']}")
    if data.get("registration_number"):
        fields.append(f"Registration: {data['registration_number']}")
    if data.get("value") or data.get("price"):
        value = data.get("value") or data.get("price")
        fields.append(f"Value: {value}")
    
    return " | ".join(fields) if fields else "Property Document"


def _extract_text_generic(data: Dict[str, Any]) -> str:
    """
    Extract searchable text from any generic document.
    Recursively extracts all string values from the JSON.
    """
    fields = []
    
    def extract_values(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.lower() in ['doc_id', 'doctype', 'timestamp', 'raw_text']:
                    continue  # Skip metadata fields
                new_prefix = f"{prefix}{key}: " if prefix else f"{key}: "
                extract_values(value, new_prefix)
        elif isinstance(obj, list):
            for item in obj:
                extract_values(item, prefix)
        elif obj is not None and str(obj).strip():
            fields.append(f"{prefix}{obj}" if prefix else str(obj))
    
    extract_values(data)
    return " | ".join(fields[:20]) if fields else "Document"  # Limit to 20 fields


def _build_searchable_text(doc_json: Dict[str, Any]) -> str:
    """
    Build a clean, searchable text representation from document JSON.
    Routes to appropriate extractor based on document type.
    
    Args:
        doc_json: The document JSON containing DOCtype and extracted data
        
    Returns:
        A merged clean text string optimized for embedding
    """
    doc_type = doc_json.get("DOCtype", "").upper()
    
    # Get the extracted data (could be nested under 'data' or 'extracted_data' key)
    data = doc_json.get("data") or doc_json.get("extracted_data") or doc_json
    
    # Route to appropriate extractor
    extractors = {
        "CNIC": _extract_text_cnic,
        "EMR": _extract_text_emr,
        "ERP": _extract_text_erp,
        "INVOICE": _extract_text_erp,
        "PROPERTY": _extract_text_property,
    }
    
    extractor = extractors.get(doc_type, _extract_text_generic)
    extracted_text = extractor(data)
    
    # Prepend document type for better context
    full_text = f"[{doc_type}] {extracted_text}" if doc_type else extracted_text
    
    return full_text


def _generate_embedding(text: str) -> np.ndarray:
    """
    Generate normalized embedding vector for the given text.
    
    Args:
        text: The text to embed
        
    Returns:
        Normalized embedding vector as numpy array
    """
    model = _get_model()
    
    # Generate embedding
    embedding = model.encode(text, convert_to_numpy=True)
    
    # Normalize for cosine similarity (FAISS IndexFlatIP)
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding.astype(np.float32)


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================

def add_document_to_index(doc_json: Dict[str, Any]) -> str:
    """
    Add a document to the FAISS index.
    
    This function takes the output JSON from the existing OCR/extraction pipeline,
    generates an embedding from the key fields, and stores it in the FAISS index
    along with metadata for retrieval.
    
    Args:
        doc_json: The document JSON from the extraction pipeline.
                  Must contain 'DOCtype' and extracted fields.
                  
    Returns:
        The doc_id (UUID) assigned to this document.
        
    Example:
        >>> doc = {
        ...     "DOCtype": "CNIC",
        ...     "name": "Ali Khan",
        ...     "father_name": "Ahmed Khan",
        ...     "identity_number": "12345-1234567-1"
        ... }
        >>> doc_id = add_document_to_index(doc)
        >>> print(doc_id)
        'a1b2c3d4-e5f6-7890-abcd-ef1234567890'
    """
    global _metadata
    
    # Get or create index
    index = _get_index()
    
    # Generate or use existing doc_id
    doc_id = doc_json.get("doc_id") or str(uuid.uuid4())
    
    # Extract document type
    doc_type = doc_json.get("DOCtype", "UNKNOWN")
    
    # Build searchable text from document
    searchable_text = _build_searchable_text(doc_json)
    print(f"[Embedding Pipeline] Indexing {doc_type} document: {doc_id[:8]}...")
    print(f"[Embedding Pipeline] Searchable text: {searchable_text[:100]}...")
    
    # Generate embedding
    embedding = _generate_embedding(searchable_text)
    
    # Add to FAISS index
    embedding_2d = embedding.reshape(1, -1)
    index.add(embedding_2d)
    
    # Store metadata (exclude raw_text to save space, keep essential fields)
    metadata_entry = {
        "doc_id": doc_id,
        "DOCtype": doc_type,
        "indexed_at": datetime.now().isoformat(),
        "searchable_text": searchable_text,
        "metadata": _extract_metadata_fields(doc_json)
    }
    _metadata.append(metadata_entry)
    
    # Persist to disk
    _save_index()
    
    print(f"[Embedding Pipeline] Document indexed successfully. Total documents: {len(_metadata)}")
    
    return doc_id


def _extract_metadata_fields(doc_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key metadata fields from document for search results.
    Excludes large fields like raw_text.
    """
    excluded_keys = {'raw_text', 'ocr_text', 'full_text', 'image_data', 'base64'}
    
    data = doc_json.get("data") or doc_json.get("extracted_data") or doc_json
    
    metadata = {}
    for key, value in data.items():
        if key.lower() in excluded_keys:
            continue
        if key in ['DOCtype', 'doc_id']:
            continue
        # Only include simple types and short lists
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        elif isinstance(value, list) and len(value) <= 10:
            metadata[key] = value
        elif isinstance(value, dict) and len(str(value)) < 500:
            metadata[key] = value
    
    return metadata


def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform semantic search across all indexed documents.
    
    This function takes a natural language query, generates its embedding,
    and finds the most similar documents in the FAISS index.
    
    Args:
        query: Natural language search query
               (e.g., "Find CNIC for Ali Khan", "Medical records with diabetes")
        top_k: Number of top results to return (default: 5)
        
    Returns:
        List of matching documents with scores and metadata.
        
    Example:
        >>> results = semantic_search("Find invoice from ABC Company", top_k=3)
        >>> for result in results:
        ...     print(f"{result['DOCtype']}: {result['score']:.2f}")
        
    Output Format:
        [
            {
                "doc_id": "uuid",
                "DOCtype": "CNIC",
                "score": 0.82,
                "metadata": {
                    "name": "Ali Khan",
                    "identity_number": "12345-1234567-1"
                }
            }
        ]
    """
    global _metadata
    
    # Get index
    index = _get_index()
    
    # Check if index is empty
    if index.ntotal == 0:
        print("[Embedding Pipeline] Warning: Index is empty. No documents to search.")
        return []
    
    print(f"[Embedding Pipeline] Searching for: '{query}' (top {top_k})")
    
    # Generate query embedding
    query_embedding = _generate_embedding(query)
    query_embedding_2d = query_embedding.reshape(1, -1)
    
    # Adjust top_k if we have fewer documents
    actual_top_k = min(top_k, index.ntotal)
    
    # Search FAISS index
    scores, indices = index.search(query_embedding_2d, actual_top_k)
    
    # Build results
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0 or idx >= len(_metadata):
            continue  # Skip invalid indices
            
        doc_metadata = _metadata[idx]
        
        result = {
            "doc_id": doc_metadata["doc_id"],
            "DOCtype": doc_metadata["DOCtype"],
            "score": round(float(score), 4),
            "metadata": doc_metadata["metadata"]
        }
        results.append(result)
    
    print(f"[Embedding Pipeline] Found {len(results)} results")
    
    return results


def get_index_stats() -> Dict[str, Any]:
    """
    Get statistics about the current FAISS index.
    
    Returns:
        Dictionary containing index statistics.
    """
    index = _get_index()
    
    # Count documents by type
    doc_type_counts = {}
    for entry in _metadata:
        doc_type = entry.get("DOCtype", "UNKNOWN")
        doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
    
    return {
        "total_documents": index.ntotal,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "model_name": EMBEDDING_MODEL_NAME,
        "index_file": FAISS_INDEX_PATH,
        "metadata_file": METADATA_PATH,
        "documents_by_type": doc_type_counts
    }


def clear_index() -> None:
    """
    Clear the entire FAISS index and metadata.
    Use with caution - this deletes all indexed documents.
    """
    global _index, _metadata
    
    print("[Embedding Pipeline] Clearing index...")
    
    # Create fresh index
    _index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    _metadata = []
    
    # Save empty index
    _save_index()
    
    print("[Embedding Pipeline] Index cleared successfully")


def rebuild_index_from_json(json_file_path: str) -> int:
    """
    Rebuild the FAISS index from a JSON file containing documents.
    
    Args:
        json_file_path: Path to a JSON file containing document array
        
    Returns:
        Number of documents indexed
    """
    print(f"[Embedding Pipeline] Rebuilding index from: {json_file_path}")
    
    # Clear existing index
    clear_index()
    
    # Load documents
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both array and single document
    documents = data if isinstance(data, list) else [data]
    
    # Index each document
    count = 0
    for doc in documents:
        if isinstance(doc, dict) and doc.get("DOCtype"):
            add_document_to_index(doc)
            count += 1
    
    print(f"[Embedding Pipeline] Rebuilt index with {count} documents")
    return count


# =============================================================================
# CONVENIENCE FUNCTIONS FOR TESTING
# =============================================================================

def search_by_type(query: str, doc_type: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for documents of a specific type.
    
    Args:
        query: Natural language search query
        doc_type: Document type filter (e.g., "CNIC", "EMR", "ERP")
        top_k: Number of results before filtering
        
    Returns:
        Filtered list of matching documents
    """
    # Get more results to account for filtering
    results = semantic_search(query, top_k=top_k * 3)
    
    # Filter by document type
    filtered = [r for r in results if r["DOCtype"].upper() == doc_type.upper()]
    
    return filtered[:top_k]


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    # Demo/Test code
    print("=" * 60)
    print("Embedding Pipeline - Demo")
    print("=" * 60)
    
    # Example CNIC document
    sample_cnic = {
        "DOCtype": "CNIC",
        "name": "Ali Khan",
        "father_name": "Ahmed Khan",
        "identity_number": "12345-1234567-1",
        "date_of_birth": "1990-05-15",
        "date_of_expiry": "2030-05-15"
    }
    
    # Example EMR document
    sample_emr = {
        "DOCtype": "EMR",
        "patient_name": "Fatima Ahmed",
        "doctor_name": "Dr. Hassan",
        "diagnosis": ["Diabetes Type 2", "Hypertension"],
        "medicines": ["Metformin 500mg", "Lisinopril 10mg"],
        "date": "2025-12-01"
    }
    
    # Example Invoice
    sample_invoice = {
        "DOCtype": "ERP",
        "invoice_number": "INV-2025-001",
        "vendor_name": "ABC Electronics",
        "items": ["Laptop", "Mouse", "Keyboard"],
        "total_amount": "125,000 PKR",
        "date": "2025-12-15"
    }
    
    print("\n1. Adding sample documents to index...")
    add_document_to_index(sample_cnic)
    add_document_to_index(sample_emr)
    add_document_to_index(sample_invoice)
    
    print("\n2. Index statistics:")
    stats = get_index_stats()
    print(json.dumps(stats, indent=2))
    
    print("\n3. Semantic search: 'Find CNIC for Ali'")
    results = semantic_search("Find CNIC for Ali", top_k=3)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    print("\n4. Semantic search: 'diabetes medication'")
    results = semantic_search("diabetes medication", top_k=3)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    print("\n5. Semantic search: 'electronics invoice'")
    results = semantic_search("electronics invoice", top_k=3)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
