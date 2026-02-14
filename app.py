"""
app.py — MJPEG streaming server for Raspberry Pi 4B + Arducam IMX708.

Architecture
────────────
  VideoCapture (CSI / /dev/video0)
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

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0          # /dev/video0 — CSI camera via V4L2
JPEG_QUALITY = 80         # 1-100; lower = smaller frames, higher = sharper
FRAME_WIDTH  = 1280       # requested capture width  (camera may clamp)
FRAME_HEIGHT = 720        # requested capture height (camera may clamp)

# ──────────────────────────────────────────────────────────────────────────────
# Camera initialisation
# ──────────────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        f"Cannot open camera at index {CAMERA_INDEX}. "
        "Is the CSI ribbon seated and the camera enabled in raspi-config?"
    )

# NOTE: We intentionally do NOT call cap.set() for FRAME_WIDTH / FRAME_HEIGHT.
# On Raspberry Pi OS Bookworm the libcamerify wrapper passes resolution as a
# JSON array internally; OpenCV's cap.set()/cap.get() trigger a fatal C++
# assertion ('!isArray_' failed) when they encounter that array type.
# Instead we let libcamera dictate the native hardware resolution.
print(f"[camera] opened /dev/video{CAMERA_INDEX}  "
      f"(native resolution — controlled by libcamerify)")


def _release_camera() -> None:
    """Release the V4L2 device so the kernel drops its lock on the CSI camera."""
    if cap.isOpened():
        cap.release()
        print("[camera] released")


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
        Raw BGR image captured by OpenCV (H×W×3, dtype uint8).

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
        success, frame = cap.read()

        if not success:
            # Transient read failure — skip this tick and retry.
            # On a Pi this occasionally happens during thermal throttling.
            continue

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
