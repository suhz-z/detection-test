# Vehicle Detection & Speed Estimation (Demo)

This demo detects vehicles using YOLOv8, tracks them with a lightweight SORT-like tracker,
maps pixel coordinates to real-world meters via homography, and estimates speed in km/h.

## Contents
- `main.py` - entry point (configured to use `test_video.mp4`)
- `detector.py` - YOLOv8 wrapper
- `tracker.py` - lightweight tracker and history for speed
- `homography.py` - pixel -> world mapping helper
- `utils.py` - drawing helpers
- `sample_calibration.json` - example calibration (replace with measured points)
- `requirements.txt` - needed Python packages

## Setup
1. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your test video into the project folder and name it `test_video.mp4`
   (or edit `VIDEO_SOURCE` in `main.py` to match your filename / RTSP URL).

4. Edit `sample_calibration.json` with at least 4 non-collinear correspondences:
   - `image_points`: pixel coordinates from your image
   - `world_points`: matching coordinates in meters on the road plane
   Accurate calibration is crucial for correct speed (km/h) outputs.

5. Run:
   ```bash
   python main.py
   ```

## Notes & tips
- If your camera has lens distortion, undistort frames before running the pipeline.
- For production use, replace the tracker with DeepSORT/ByteTrack for better ID persistence.
- Use exact per-frame timestamps for dt when using live camera feeds for best speed estimates.

