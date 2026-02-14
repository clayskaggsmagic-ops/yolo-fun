"""
app.py — MJPEG streaming server for Raspberry Pi 4B + Arducam IMX708.

Uses the native Picamera2 library for frame capture (bypasses the broken
libcamerify + V4L2 pipeline) and OpenCV only for JPEG compression.

Architecture
────────────
  Picamera2  (CSI → ISP → BGR888 numpy array)
       │
       ▼
  generate_frames()  ──▶  process_frame(frame)   ← YOLO hook (no-op for now)
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
    Process a single BGR frame and return the (possibly modified) frame.

    This is the injection point for the YOLO11-nano ncnn inference step.
    For now it passes the frame through untouched.

    Parameters
    ----------
    frame : numpy.ndarray
        Raw BGR image captured by Picamera2 (H×W×3, dtype uint8).

    Returns
    -------
    numpy.ndarray
        The frame to be JPEG-encoded and streamed to the client.
    """
    # ── Future: YOLO inference goes here ──────────────────────────────────
    # results = model(frame)
    # frame   = results[0].plot()          # draw bounding boxes
    # ──────────────────────────────────────────────────────────────────────
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
