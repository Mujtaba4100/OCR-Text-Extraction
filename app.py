import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file
from ai_pipeline import process

# Import embedding pipeline (graceful fallback if unavailable)
try:
    from embedding_pipeline import add_document_to_index
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
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if uploaded_file and uploaded_file.filename != "":
           
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
                    # Extraction continues even if embedding fails

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(extracted_data, f, indent=4, ensure_ascii=False)

            
            return redirect(url_for("result", json_file=json_file))

    return render_template("index.html", extracted_data=None)


@app.route("/result", methods=["GET"])
def result():
    json_file = request.args.get("json_file")
    data = None

    if json_file:
        json_path = os.path.join(DATA_FOLDER, json_file)
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

    return render_template("index.html", extracted_data=data)

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
