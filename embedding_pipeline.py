"""
Embedding Pipeline Module for AI Document Management System
============================================================
Provides embedding generation and semantic search using Sentence Transformers and FAISS.

Public API:
    add_document_to_index(doc_json: dict) -> str
    semantic_search(query: str, top_k: int = 5) -> list
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

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

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
    """Lazy load the SentenceTransformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _get_index() -> faiss.IndexFlatIP:
    """Get or create the FAISS index. Loads from disk if exists."""
    global _index, _metadata
    
    if _index is None:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
            _index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                _metadata = json.load(f)
        else:
            _index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            _metadata = []
    
    return _index


def _save_index() -> None:
    """Persist the FAISS index and metadata to disk."""
    global _index, _metadata
    
    if _index is not None:
        faiss.write_index(_index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(_metadata, f, indent=2, ensure_ascii=False)


# =============================================================================
# TEXT EXTRACTION (Generic)
# =============================================================================

def _build_searchable_text(doc_json: Dict[str, Any]) -> str:
    """
    Build searchable text from any document JSON.
    Recursively extracts all string values, excluding metadata fields.
    """
    excluded_keys = {'doc_id', 'rawtext', 'raw_text', 'ocr_text', 'image_data', 'base64', 'timestamp'}
    fields = []
    
    def extract_values(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.lower() in excluded_keys:
                    continue
                extract_values(value, f"{key}: ")
        elif isinstance(obj, list):
            for item in obj:
                extract_values(item, prefix)
        elif obj is not None and str(obj).strip():
            fields.append(f"{prefix}{obj}" if prefix else str(obj))
    
    extract_values(doc_json)
    return " | ".join(fields) if fields else "Document"


def _generate_embedding(text: str) -> np.ndarray:
    """Generate normalized embedding vector for the given text."""
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.astype(np.float32)


# =============================================================================
# PUBLIC API
# =============================================================================

def add_document_to_index(doc_json: Dict[str, Any]) -> str:
    """
    Add a document to the FAISS index.
    
    Takes the output JSON from the extraction pipeline, generates an embedding
    from the key fields, and stores it in the FAISS index with metadata.
    
    Args:
        doc_json: Document JSON from ai_pipeline.process().
                  Should contain 'DOCtype' and extracted fields.
                  
    Returns:
        The doc_id (UUID) assigned to this document.
    """
    global _metadata
    
    index = _get_index()
    
    doc_id = doc_json.get("doc_id") or str(uuid.uuid4())
    doc_type = doc_json.get("DOCtype", "UNKNOWN")
    
    # Build searchable text and generate embedding
    searchable_text = _build_searchable_text(doc_json)
    embedding = _generate_embedding(searchable_text)
    
    # Add to FAISS index
    index.add(embedding.reshape(1, -1))
    
    # Store metadata (exclude large fields)
    metadata_entry = {
        "doc_id": doc_id,
        "DOCtype": doc_type,
        "indexed_at": datetime.now().isoformat(),
        "metadata": {k: v for k, v in doc_json.items() 
                     if k not in ('rawtext', 'raw_text', 'doc_id', 'DOCtype')
                     and isinstance(v, (str, int, float, bool, list, dict))}
    }
    _metadata.append(metadata_entry)
    
    _save_index()
    return doc_id


def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform semantic search across all indexed documents.
    
    Args:
        query: Natural language search query.
        top_k: Number of top results to return (default: 5).
        
    Returns:
        List of matching documents:
        [
            {
                "doc_id": "uuid",
                "DOCtype": "CNIC",
                "score": 0.82,
                "metadata": {"name": "Ali Khan", ...}
            }
        ]
    """
    global _metadata
    
    index = _get_index()
    
    if index.ntotal == 0:
        return []
    
    # Generate query embedding and search
    query_embedding = _generate_embedding(query)
    actual_top_k = min(top_k, index.ntotal)
    scores, indices = index.search(query_embedding.reshape(1, -1), actual_top_k)
    
    # Build results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(_metadata):
            doc = _metadata[idx]
            results.append({
                "doc_id": doc["doc_id"],
                "DOCtype": doc["DOCtype"],
                "score": round(float(score), 4),
                "metadata": doc["metadata"]
            })
    
    return results
