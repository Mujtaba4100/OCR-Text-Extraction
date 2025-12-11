# app.py
import os
import json
from flask import Flask, render_template, request, redirect, url_for, send_file
from ai_pipeline import process

app = Flask(__name__, template_folder="templates")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    extracted_data = None
    filename = None
    json_file = "data.json"
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if uploaded_file and uploaded_file.filename != "":
            filename = uploaded_file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            uploaded_file.save(file_path)

            try:
                extracted_data = process(file_path)
            except Exception as e:
                extracted_data = {"error": str(e)}
            # save JSON
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(extracted_data, f, indent=4, ensure_ascii=False)

            return redirect(url_for("result"))
    return render_template("index.html", extracted_data=None)

@app.route("/result")
def result():
    json_file = "data.json"
    data = None
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    return render_template("index.html", extracted_data=data)

@app.route("/download")
def download_json():
    json_file = "data.json"
    if os.path.exists(json_file):
        return send_file(json_file, as_attachment=True)
    return "No data", 404

if __name__ == "__main__":
    app.run(debug=True, port=5000)
