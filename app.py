import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file
from ai_pipeline import process
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("api_key"))

# Import embedding pipeline (graceful fallback if unavailable)
try:
    from embedding_pipeline import add_document_to_index, semantic_search
    EMBEDDING_ENABLED = True
except ImportError:
    EMBEDDING_ENABLED = False
    logging.warning("[App] Embedding pipeline not available. Semantic search disabled.")

app = Flask(__name__, template_folder="templates")

# Configure logging for embedding errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def answer_query_with_llm(user_query: str, search_results: list) -> str:
    """
    Generate a natural language answer using Gemini based on semantic search results.
    
    Args:
        user_query: The user's natural language question
        search_results: List of top matching documents from semantic search
        
    Returns:
        A concise natural language answer or "Information not available."
    """
    if not search_results:
        return "Information not available. No matching documents found."
    
    # Build context from search results metadata
    context_parts = []
    for i, result in enumerate(search_results[:3], 1):
        doc_type = result.get("DOCtype", "Unknown")
        metadata = result.get("metadata", {})
        
        # Format metadata as readable text
        fields = []
        for key, value in metadata.items():
            if value and key not in ('rawtext', 'raw_text'):
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                fields.append(f"{key.replace('_', ' ').title()}: {value}")
        
        if fields:
            context_parts.append(f"Document {i} ({doc_type}):\n" + "\n".join(fields))
    
    if not context_parts:
        return "Information not available."
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Based on the following document information, answer the user's question concisely.
If the information is not available in the documents, respond with "Information not available."

Document Information:
{context}

User Question: {user_query}

Provide a direct, concise answer:"""
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        answer = response.text.strip()
        return answer if answer else "Information not available."
    except Exception as e:
        logging.error(f"[LLM] Error generating answer: {e}")
        return "Unable to generate answer. Please try again."

UPLOAD_FOLDER = "uploads"
DATA_FOLDER = "data"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    extracted_data = None
    json_file = None
    search_results = None
    search_query = ""
    llm_answer = None
    
    if request.method == "POST":
        # Handle document upload
        if "file" in request.files and request.files["file"].filename != "":
            uploaded_file = request.files["file"]
            filename = uploaded_file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            uploaded_file.save(file_path)

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            json_file = f"data_{timestamp}.json"
            json_path = os.path.join(DATA_FOLDER, json_file)

            try:
                extracted_data = process(file_path)
            except Exception as e:
                extracted_data = {"error": str(e)}

            # Index document for semantic search (non-blocking, error-safe)
            if EMBEDDING_ENABLED and "error" not in extracted_data:
                try:
                    add_document_to_index(extracted_data)
                    logging.info(f"[Embedding] Document indexed: {filename}")
                except Exception as e:
                    logging.error(f"[Embedding] Failed to index {filename}: {e}")

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(extracted_data, f, indent=4, ensure_ascii=False)
        
        # Handle semantic search
        elif "search_query" in request.form:
            search_query = request.form.get("search_query", "").strip()
            if search_query and EMBEDDING_ENABLED:
                try:
                    search_results = semantic_search(search_query, top_k=3)
                    logging.info(f"[Search] Query: '{search_query}' returned {len(search_results)} results")
                    
                    # Generate LLM answer from search results
                    llm_answer = answer_query_with_llm(search_query, search_results)
                    logging.info(f"[LLM] Generated answer for query: '{search_query}'")
                except Exception as e:
                    logging.error(f"[Search] Failed: {e}")
                    search_results = []
                    llm_answer = "An error occurred while searching. Please try again."

    return render_template("index.html", 
                          extracted_data=extracted_data, 
                          json_file=json_file,
                          search_results=search_results,
                          search_query=search_query,
                          llm_answer=llm_answer,
                          embedding_enabled=EMBEDDING_ENABLED)


@app.route("/result", methods=["GET"])
def result():
    json_file = request.args.get("json_file")
    data = None

    if json_file:
        json_path = os.path.join(DATA_FOLDER, json_file)
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

    return render_template("index.html", 
                          extracted_data=data, 
                          json_file=json_file,
                          search_results=None,
                          search_query="",
                          llm_answer=None,
                          embedding_enabled=EMBEDDING_ENABLED)

@app.route("/download", methods=["GET"])
def download_json():
    json_file = request.args.get("json_file")
    if json_file:
        json_path = os.path.join(DATA_FOLDER, json_file)
        if os.path.exists(json_path):
            return send_file(json_path, as_attachment=True)

    return "No data available", 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)
