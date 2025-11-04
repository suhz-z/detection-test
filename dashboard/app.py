from flask import Flask, render_template, request, send_file
import requests
import os


app = Flask(__name__)
FASTAPI_URL = "http://localhost:8001"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "No file uploaded."
        files = {"file": (file.filename, file.read())}
        r = requests.post(f"{FASTAPI_URL}/upload_video/", files=files)

        zip_path = os.path.join(RESULTS_DIR, "results.zip")
        with open(zip_path, "wb") as f:
            f.write(r.content)
        return send_file(zip_path, as_attachment=True)
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
