from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse
from datetime import datetime
from main import process_video
import tempfile, os

app = FastAPI(title="Vehicle Detection API")

@app.get("/")
def home():
    return {"status": "ok", "message": "YOLOv8 + Supervision Backend Ready"}

@app.post("/upload_video/")
async def upload_video(
    file: UploadFile = File(...),
    timestamp: str = Query(None, description="Shared timestamp from Flask"),
):
    # Generate timestamp if not provided
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Define paths
    tmp_path = os.path.join(output_dir, f"input_{timestamp}.mp4")
    output_path = os.path.join(output_dir, f"annotated_{timestamp}.mp4")

    # Save uploaded file
    with open(tmp_path, "wb") as tmp:
        tmp.write(await file.read())

    # Run vehicle detection and annotation
    process_video(tmp_path, output_path)

    # Return video directly to client
    return FileResponse(
        output_path,
        filename=f"annotated_{timestamp}.mp4",
        media_type="video/mp4"
    )
