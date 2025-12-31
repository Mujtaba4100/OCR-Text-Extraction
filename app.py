import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file
from ai_pipeline import process

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
                    search_results = semantic_search(search_query, top_k=5)
                    logging.info(f"[Search] Query: '{search_query}' returned {len(search_results)} results")
                except Exception as e:
                    logging.error(f"[Search] Failed: {e}")
                    search_results = []

    return render_template("index.html", 
                          extracted_data=extracted_data, 
                          json_file=json_file,
                          search_results=search_results,
                          search_query=search_query,
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
