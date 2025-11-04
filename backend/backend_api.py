from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from main import process_video
import tempfile, zipfile, os

app = FastAPI(title="Vehicle Detection API")

@app.get("/")
def home():
    return {"status": "ok", "message": "YOLOv8 + Supervision Backend Ready"}

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    output_path = "data/output_annotated.mp4"
    csv_path = "data/vehicle_log.csv"
    os.makedirs("data", exist_ok=True)

    process_video(tmp_path, output_path, csv_path)

    # Zip both files
    zip_path = "data/results.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(output_path, os.path.basename(output_path))
        zf.write(csv_path, os.path.basename(csv_path))

    return FileResponse(zip_path, filename="results.zip", media_type="application/zip")
