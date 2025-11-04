"""
app.py — Simple Flask web dashboard for live vehicle detection
Streams annotated frames from main.process_frame() and allows CSV download.
"""

from flask import Flask, render_template, Response, send_file
from main import process_frame
import os

app = Flask(__name__)
CSV_PATH = 'data/vehicle_log.csv'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    # Live video stream
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/download_csv')
def download_csv():
    # Download the latest CSV
    if not os.path.exists(CSV_PATH):
        return "No CSV log yet."
    return send_file(CSV_PATH, as_attachment=True)


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
