"""
app.py — MJPEG streaming server + YOLOv8 Nano person detection
           for Raspberry Pi 4B + Arducam IMX708.

Optimised for maximum FPS on the Pi 4's Cortex-A72:
  • 640×480 capture (fewer pixels to process)
  • Threaded camera capture (decoupled from inference)
  • Skip-frame YOLO (inference every N frames, blit last overlay between)
  • Multi-core cv2.dnn inference (4 threads)
  • Lower JPEG quality for faster encoding

Architecture
────────────
  Picamera2  (CSI → ISP → RGB numpy array)
       │  [background thread]
       ▼
  CameraStream._reader()  →  self._frame  (latest raw frame)
       │
       ▼
  generate_frames()  ──▶  process_frame(frame)   ← RGB→BGR + YOLO via cv2.dnn
       │                                             (runs every Nth frame)
       ▼
  cv2.imencode(".jpg")
       │
       ▼
  HTTP multipart/x-mixed-replace  ──▶  Browser <img> tag
"""

import atexit
import threading

import cv2
import numpy as np
from flask import Flask, Response, render_template
from picamera2 import Picamera2

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
JPEG_QUALITY      = 60        # lower = smaller/faster, 60 is fine for streaming
FRAME_WIDTH       = 640       # capture width  (smaller = faster everything)
FRAME_HEIGHT      = 480       # capture height
MODEL_INPUT_SIZE  = 320       # YOLO input size (320 is ~4× faster than 640)
CONFIDENCE_THRESH = 0.45      # minimum confidence to keep a detection
NMS_THRESH        = 0.40      # IoU threshold for Non-Maximum Suppression
PERSON_CLASS_ID   = 0         # COCO class 0 = person
YOLO_EVERY_N      = 3         # run YOLO every Nth frame (show last overlay between)

# Tell OpenCV's DNN to use all 4 CPU cores for inference
cv2.setNumThreads(4)

# ──────────────────────────────────────────────────────────────────────────────
# Threaded camera capture
# ──────────────────────────────────────────────────────────────────────────────


class CameraStream:
    """
    Captures frames in a background thread so the main loop never waits
    for the camera.  Always holds the *latest* frame; old frames are
    silently dropped.
    """

    def __init__(self, width, height):
        self._picam2 = Picamera2()
        config = self._picam2.create_video_configuration(
            main={"format": "BGR888", "size": (width, height)}
        )
        self._picam2.configure(config)
        self._picam2.start()

        self._frame = None
        self._lock = threading.Lock()
        self._running = True

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

        print(f"[camera] Picamera2 started  "
              f"resolution={width}×{height}  format=BGR888  (threaded)")

    def _reader(self):
        """Continuously grab frames in a background thread."""
        while self._running:
            frame = self._picam2.capture_array()
            with self._lock:
                self._frame = frame

    def read(self):
        """Return the most recent frame (or None if not ready yet)."""
        with self._lock:
            return self._frame

    def stop(self):
        """Stop the capture thread and release the hardware."""
        self._running = False
        self._thread.join(timeout=2)
        try:
            self._picam2.stop()
            self._picam2.close()
            print("[camera] released")
        except Exception:
            pass


cam = CameraStream(FRAME_WIDTH, FRAME_HEIGHT)

# ──────────────────────────────────────────────────────────────────────────────
# YOLO model  (loaded once at startup via OpenCV's DNN module)
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "yolov8n.onnx"

net = cv2.dnn.readNetFromONNX(MODEL_PATH)
print(f"[yolo]   loaded {MODEL_PATH}  (cv2.dnn / ONNX, input={MODEL_INPUT_SIZE})")


# ──────────────────────────────────────────────────────────────────────────────
# Hardware lifecycle — release the CSI camera on exit
# ──────────────────────────────────────────────────────────────────────────────

def _release_camera() -> None:
    """Stop the threaded camera so the kernel frees the device."""
    cam.stop()


atexit.register(_release_camera)


# ──────────────────────────────────────────────────────────────────────────────
# Frame processing pipeline
# ──────────────────────────────────────────────────────────────────────────────

# Pre-compute scale factors (MODEL_INPUT_SIZE → native resolution).
_sx = FRAME_WIDTH  / MODEL_INPUT_SIZE
_sy = FRAME_HEIGHT / MODEL_INPUT_SIZE


def process_frame(frame):
    """
    Fix colour channels, run YOLOv8n via cv2.dnn, and annotate persons.

    Parameters
    ----------
    frame : numpy.ndarray
        Raw RGB image from Picamera2 (H×W×3, dtype uint8).

    Returns
    -------
    numpy.ndarray
        Annotated BGR frame.
    """
    # ── 1. Colour fix ────────────────────────────────────────────────────
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # ── 2. Blob creation (320×320 is ~4× faster than 640×640) ────────────
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        swapRB=False, crop=False,
    )

    # ── 3. Forward pass ──────────────────────────────────────────────────
    net.setInput(blob)
    outputs = net.forward()          # shape: [1, 84, 8400]

    # ── 4. Tensor parsing ────────────────────────────────────────────────
    preds = outputs[0].T             # shape: [8400, 84]

    # ── 5-6. Filter + NMS ────────────────────────────────────────────────
    boxes = []
    confidences = []

    for row in preds:
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        conf = float(class_scores[class_id])

        if class_id != PERSON_CLASS_ID or conf < CONFIDENCE_THRESH:
            continue

        cx, cy, w, h = row[0], row[1], row[2], row[3]
        x = int((cx - w / 2) * _sx)
        y = int((cy - h / 2) * _sy)
        w = int(w * _sx)
        h = int(h * _sy)

        boxes.append([x, y, w, h])
        confidences.append(conf)

    indices = cv2.dnn.NMSBoxes(boxes, confidences,
                               CONFIDENCE_THRESH, NMS_THRESH)

    # ── 7. Annotation ───────────────────────────────────────────────────
    for i in indices:
        idx = int(i) if isinstance(i, (int, np.integer)) else int(i[0])
        x, y, w, h = boxes[idx]
        conf = confidences[idx]

        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      color=(0, 255, 0), thickness=2)

        label = f"Person {conf:.0%}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2,
        )
        cv2.rectangle(frame, (x, y - th - 10), (x + tw, y),
                      color=(0, 255, 0), thickness=-1)
        cv2.putText(frame, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2)

    return frame


# ──────────────────────────────────────────────────────────────────────────────
# MJPEG generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_frames():
    """
    Infinite generator that yields MJPEG-formatted byte chunks.

    Runs YOLO only every YOLO_EVERY_N frames.  Between inference frames,
    it re-uses the last annotated frame — so the stream stays smooth even
    though the boxes update less frequently.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    frame_count = 0
    last_annotated = None

    while True:
        frame = cam.read()
        if frame is None:
            continue

        frame_count += 1

        if frame_count % YOLO_EVERY_N == 0 or last_annotated is None:
            # Run the full YOLO pipeline
            last_annotated = process_frame(frame)
        else:
            # Just colour-fix and show (no YOLO this tick)
            last_annotated = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ok, buffer = cv2.imencode(".jpg", last_annotated, encode_params)
        if not ok:
            continue

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
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        _release_camera()
