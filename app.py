import os
from flask import Flask, render_template, request
from ai_pipeline import process

app = Flask(__name__)
upload_folder = "uploads"
os.makedirs(upload_folder, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    extracted_text = ""
    file_path = None

    if request.method == "POST":
        uploaded_file = request.files["file"]
        if uploaded_file.filename != "":
            file_path = os.path.join(upload_folder, uploaded_file.filename)
            uploaded_file.save(file_path)

        extracted_text = process(file_path)

    return render_template("index.html", extracted_text=extracted_text, file_path=file_path)

if __name__ == "__main__":
    app.run(debug=True)
