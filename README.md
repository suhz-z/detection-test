# detection-test — simple README

A small demo that uses YOLOv8 (Ultralytics) + Supervision to detect vehicles, annotate video,
The repo contains a FastAPI backend endpoint and a small Flask dashboard.

Quick steps (Windows):

1) Create & activate a virtual env:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

Notes about torch: the `requirements.txt` lists `torch` without a pinned CUDA build.
If you need a specific CUDA-enabled wheel, install it from the official PyTorch instructions
before or after installing the other packages.

3) Run the backend API (FastAPI) — the dashboard expects the API on port 8001:

```powershell
uvicorn backend.backend_api:app --reload --port 8001
```

4) Run the dashboard (Flask) which provides a simple upload UI:

```powershell
python -m dashboard.app
```

Files of interest:
- `backend/main.py` — main detection pipeline and helper calls
- `backend/detector.py` — YOLOv8 wrapper
- `backend/homography.py` — homography / calibration helper
- `backend/utils.py` — drawing & helper functions
- `backend/backend_api.py` — FastAPI endpoint used to upload video and return results.zip
- `dashboard/app.py` — Flask UI that uploads to the FastAPI backend

Quick verification:
- Upload a short MP4 via the dashboard or POST to `/upload_video/` on the FastAPI server.
- The API will return `results.zip` containing an annotated MP4

Notes: 
- To Add live camera feed change the video capture to the feed http/rtsp

