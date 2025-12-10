import os
import json
from flask import Flask, render_template, request
from ai_pipeline import process

app = Flask(__name__)
upload_folder = "uploads"
os.makedirs(upload_folder, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    extracted_data = None
    file_path = None
    json_file = "data.json"

    if request.method == "POST":
        uploaded_file = request.files["file"]
        if uploaded_file.filename != "":
            file_path = os.path.join(upload_folder, uploaded_file.filename)
            uploaded_file.save(file_path)

            extracted_data = process(file_path)

            # Save JSON (overwrite or append logic)
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(extracted_data, f, indent=4, ensure_ascii=False)

    return render_template("index.html", extracted_data=extracted_data)

if __name__ == "__main__":
    app.run(debug=True)
