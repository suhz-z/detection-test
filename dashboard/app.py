from flask import Flask, render_template, request, send_file
import requests

app = Flask(__name__)
FASTAPI_URL = "http://localhost:8001"

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
        open("results.zip", "wb").write(r.content)
        return send_file("results.zip", as_attachment=True)
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
