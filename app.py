"""
app.py — MJPEG streaming server + YOLO11 Nano person detection
           for Raspberry Pi 4B + Arducam IMX708.

Uses Picamera2 for frame capture, OpenCV's cv2.dnn module to run a
YOLO11n ONNX model (no torch/ultralytics — safe on ARMv8.0), and
Flask for MJPEG streaming.

Architecture
────────────
  Picamera2  (CSI → ISP → RGB numpy array)
       │
       ▼
  generate_frames()  ──▶  process_frame(frame)   ← RGB→BGR + YOLO via cv2.dnn
       │
       ▼
  cv2.imencode(".jpg")
       │
       ▼
  HTTP multipart/x-mixed-replace  ──▶  Browser <img> tag
"""

import atexit

import cv2
import numpy as np
from flask import Flask, Response, render_template
from picamera2 import Picamera2

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
JPEG_QUALITY      = 80        # 1-100; JPEG compression quality
FRAME_WIDTH       = 1280      # capture width  (pixels)
FRAME_HEIGHT      = 720       # capture height (pixels)
MODEL_INPUT_SIZE  = 640       # YOLO expects a 640×640 input blob
CONFIDENCE_THRESH = 0.50      # minimum confidence to keep a detection
NMS_THRESH        = 0.40      # IoU threshold for Non-Maximum Suppression
PERSON_CLASS_ID   = 0         # COCO class 0 = person

# ──────────────────────────────────────────────────────────────────────────────
# Camera initialisation  (Picamera2 — talks directly to libcamera, no V4L2)
# ──────────────────────────────────────────────────────────────────────────────
picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"format": "BGR888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
)
picam2.configure(config)
picam2.start()

print(f"[camera] Picamera2 started  "
      f"resolution={FRAME_WIDTH}×{FRAME_HEIGHT}  format=BGR888")

# ──────────────────────────────────────────────────────────────────────────────
# YOLO model  (loaded once at startup via OpenCV's DNN module)
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "yolov8n.onnx"

net = cv2.dnn.readNetFromONNX(MODEL_PATH)
print(f"[yolo]   loaded {MODEL_PATH}  (cv2.dnn / ONNX)")


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
        pass


atexit.register(_release_camera)


# ──────────────────────────────────────────────────────────────────────────────
# Frame processing pipeline
# ──────────────────────────────────────────────────────────────────────────────

# Pre-compute scale factors (640 → native resolution) once, not per-frame.
_sx = FRAME_WIDTH  / MODEL_INPUT_SIZE   # horizontal scale
_sy = FRAME_HEIGHT / MODEL_INPUT_SIZE   # vertical   scale


def process_frame(frame):
    """
    Fix colour channels, run YOLO11n via cv2.dnn, and annotate persons.

    Steps
    -----
    1. Convert RGB (Picamera2) → BGR (OpenCV).
    2. Build a 640×640 normalised blob.
    3. Forward-pass through the ONNX network.
    4. Parse the [1, 84, 8400] output tensor.
    5. Filter for class 0 (person) above CONFIDENCE_THRESH.
    6. Apply Non-Maximum Suppression to remove duplicate boxes.
    7. Scale coordinates back to 1280×720 and draw annotations.

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

    # ── 2. Blob creation ─────────────────────────────────────────────────
    #   • Scale pixel values to 0-1       (1/255.0)
    #   • Resize to 640×640               (MODEL_INPUT_SIZE)
    #   • swapRB=False — frame is already BGR
    #   • crop=False   — letterbox / stretch, don't crop
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        swapRB=False, crop=False,
    )

    # ── 3. Forward pass ──────────────────────────────────────────────────
    net.setInput(blob)
    outputs = net.forward()          # shape: [1, 84, 8400]

    # ── 4. Tensor parsing ────────────────────────────────────────────────
    #   Squeeze batch dim → [84, 8400], then transpose → [8400, 84]
    #   Each of the 8,400 rows is one candidate detection:
    #     [x_center, y_center, width, height, cls0_score, cls1_score, … cls79_score]
    preds = outputs[0].T             # shape: [8400, 84]

    # ── 5-6. Filter + NMS ────────────────────────────────────────────────
    boxes = []
    confidences = []

    for row in preds:
        # row[0:4]  = bounding box (centre-x, centre-y, w, h) in 640-space
        # row[4:84] = 80 class scores
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        conf = float(class_scores[class_id])

        if class_id != PERSON_CLASS_ID or conf < CONFIDENCE_THRESH:
            continue

        # Convert (cx, cy, w, h) in 640-space → (x, y, w, h) in native-space
        cx, cy, w, h = row[0], row[1], row[2], row[3]
        x = int((cx - w / 2) * _sx)
        y = int((cy - h / 2) * _sy)
        w = int(w * _sx)
        h = int(h * _sy)

        boxes.append([x, y, w, h])
        confidences.append(conf)

    # Non-Maximum Suppression: eliminate overlapping duplicates
    indices = cv2.dnn.NMSBoxes(boxes, confidences,
                               CONFIDENCE_THRESH, NMS_THRESH)

    # ── 7. Annotation ───────────────────────────────────────────────────
    for i in indices:
        # `i` may be an int or a 1-element array, depending on OpenCV version
        idx = int(i) if isinstance(i, (int, np.integer)) else int(i[0])
        x, y, w, h = boxes[idx]
        conf = confidences[idx]

        # Green bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      color=(0, 255, 0), thickness=2)

        # Confidence label with filled background
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
