from flask import Flask, render_template, request, send_file
import requests
import os
from datetime import datetime

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

        # Shared timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Send video to FastAPI
        files = {"file": (file.filename, file.read())}
        response = requests.post(f"{FASTAPI_URL}/upload_video/?timestamp={timestamp}", files=files)

        # Save returned annotated video
        video_filename = f"annotated_{timestamp}.mp4"
        video_path = os.path.join(RESULTS_DIR, video_filename)
        with open(video_path, "wb") as f:
            f.write(response.content)

        # Return video to user
        return send_file(video_path, as_attachment=True, download_name="annotated_video.mp4")

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
