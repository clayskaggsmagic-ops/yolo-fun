"""
app.py — MJPEG streaming server + YOLO11 Nano person detection
           for Raspberry Pi 4B + Arducam IMX708.

Uses the native Picamera2 library for frame capture (bypasses the broken
libcamerify + V4L2 pipeline), YOLO11n-ncnn for real-time person detection,
and OpenCV for annotation + JPEG compression.

Architecture
────────────
  Picamera2  (CSI → ISP → RGB numpy array)
       │
       ▼
  generate_frames()  ──▶  process_frame(frame)   ← RGB→BGR + YOLO detection
       │
       ▼
  cv2.imencode(".jpg")
       │
       ▼
  HTTP multipart/x-mixed-replace  ──▶  Browser <img> tag
"""

import atexit
import cv2
from flask import Flask, Response, render_template
from picamera2 import Picamera2
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
JPEG_QUALITY = 80         # 1-100; lower = smaller frames, higher = sharper
FRAME_WIDTH  = 1280       # capture width
FRAME_HEIGHT = 720        # capture height

# ──────────────────────────────────────────────────────────────────────────────
# Camera initialisation  (Picamera2 — talks directly to libcamera, no V4L2)
# ──────────────────────────────────────────────────────────────────────────────
picam2 = Picamera2()

# BGR888 keeps the frame natively compatible with OpenCV's colour space,
# so cv2.imencode() works without any cvtColor conversion.
config = picam2.create_video_configuration(
    main={"format": "BGR888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
)
picam2.configure(config)
picam2.start()

print(f"[camera] Picamera2 started  "
      f"resolution={FRAME_WIDTH}×{FRAME_HEIGHT}  format=BGR888")

# ──────────────────────────────────────────────────────────────────────────────
# YOLO model  (loaded once at startup to avoid per-frame memory allocation)
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH       = "yolo11n_ncnn_model"    # path to the exported ncnn model dir
CONFIDENCE_THRESH = 0.50                   # minimum confidence to draw a box
PERSON_CLASS_ID   = 0                      # COCO class 0 = person

model = YOLO(MODEL_PATH)
print(f"[yolo]   loaded {MODEL_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# Hardware lifecycle — release the CSI camera on exit
# ──────────────────────────────────────────────────────────────────────────────

def _release_camera() -> None:
    """Stop and close the Picamera2 instance so the kernel frees the device."""
    try:
        picam2.stop()
        picam2.close()
        print("[camera] released")
    except Exception:
        # If the camera was never started or already closed, ignore.
        pass


# Register the cleanup handler so the camera is freed on normal exit,
# SIGTERM, or an unhandled exception that tears down the interpreter.
atexit.register(_release_camera)


# ──────────────────────────────────────────────────────────────────────────────
# Frame processing pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_frame(frame):
    """
    Fix the colour channel order, run YOLO person detection, and annotate.

    Steps
    -----
    1. Convert RGB (Picamera2 native) → BGR (OpenCV native).
    2. Run YOLO11n-ncnn inference on the BGR frame.
    3. Filter for class 0 (person) with confidence > CONFIDENCE_THRESH.
    4. Draw bounding boxes and confidence labels.
    5. Return the annotated BGR frame (ready for cv2.imencode).

    Parameters
    ----------
    frame : numpy.ndarray
        Raw RGB image from Picamera2 (H×W×3, dtype uint8).

    Returns
    -------
    numpy.ndarray
        Annotated BGR frame.
    """
    # ── 1. Colour fix: Picamera2 outputs RGB, OpenCV expects BGR ──────────
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # ── 2. Run YOLO inference (verbose=False silences per-frame logging) ──
    results = model(frame, verbose=False)

    # ── 3–4. Filter for persons and draw boxes ────────────────────────────
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])

            # Skip anything that isn't a person or is below threshold
            if cls_id != PERSON_CLASS_ID or conf < CONFIDENCE_THRESH:
                continue

            # Bounding-box corner coordinates (top-left, bottom-right)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw the rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          color=(0, 255, 0), thickness=2)

            # Draw the confidence label above the box
            label = f"Person {conf:.0%}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1),
                          color=(0, 255, 0), thickness=-1)  # filled bg
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 2)  # black text on green bg

    return frame


# ──────────────────────────────────────────────────────────────────────────────
# MJPEG generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_frames():
    """
    Infinite generator that yields MJPEG-formatted byte chunks.

    Each chunk consists of:
        --frame\\r\\n
        Content-Type: image/jpeg\\r\\n
        \\r\\n
        <jpeg bytes>
        \\r\\n

    The browser's <img src="/video_feed"> element replaces the displayed
    image every time a new boundary arrives, producing a live video stream.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    while True:
        # Grab a BGR numpy array straight from the ISP — no V4L2 involved.
        frame = picam2.capture_array()

        # Run the frame through the processing pipeline
        frame = process_frame(frame)

        # Encode the frame as JPEG
        ok, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            continue

        # Yield the multipart boundary + JPEG payload
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Flask application
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    """Serve the viewer page with the embedded MJPEG stream."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """
    Return a streaming HTTP response whose content type tells the browser
    to treat each boundary-delimited JPEG as a replacement for the previous
    image.  This is the classic "server-push" Motion-JPEG technique.
    """
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Bind to 0.0.0.0 so the stream is reachable from any device on the LAN.
    # `threaded=True` lets Flask handle multiple browser tabs concurrently.
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        # Belt-and-suspenders: release camera even if app.run raises.
        _release_camera()
