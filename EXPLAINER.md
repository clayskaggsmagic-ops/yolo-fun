# YOLO-Fun — How Everything Works (Beginner Edition)

This guide walks through every piece of the project from scratch.
No prior Python or web-dev knowledge assumed.

---

## 1 — The Big Picture

You have a **Raspberry Pi** with a **camera** glued to it.
You want to see what the camera sees — live — on **any phone or laptop**
connected to the same Wi-Fi network.

The way we do that:

```
┌──────────────┐         ┌───────────────┐         ┌──────────────┐
│   Camera     │  ──▶    │  Python app   │  ──▶    │  Your phone  │
│  (hardware)  │  frame  │  (Flask +     │  JPEG   │  or laptop   │
│              │         │   Picamera2)  │  stream │  (browser)   │
└──────────────┘         └───────────────┘         └──────────────┘
```

1. The **camera** captures a raw image (a "frame") many times per second.
2. Our **Python script** grabs each frame, compresses it into a JPEG, and
   sends it over the network as a never-ending stream.
3. Your **browser** (`<img>` tag) keeps receiving those JPEGs and displays
   each one in place of the last — which looks like live video.

This trick is called **MJPEG** (Motion JPEG). No fancy video codecs or
WebRTC needed.

---

## 2 — The Files

```
yolo-fun/
├── app.py                 ← The main Python program (the server)
├── requirements.txt       ← Lists the Python packages we need
└── templates/
    └── index.html         ← The web page your browser loads
```

Let's go through each one.

---

## 3 — `requirements.txt`  (the shopping list)

```text
flask>=3.0,<4.0
opencv-python-headless>=4.8,<5.0
```

Think of this file like a grocery list.  Before running the app you say:

```bash
pip install -r requirements.txt
```

…and Python downloads two things:

| Package | What it does |
|---|---|
| **Flask** | A tiny web server framework. It lets us say *"when someone visits `/`, show them this HTML page"*. |
| **opencv-python-headless** | A computer-vision library. We only use it here to compress images into JPEG format. The `-headless` version skips GUI stuff the Pi doesn't need. |

> **What about Picamera2?** It's already installed on Raspberry Pi OS.
> You don't pip-install it — it comes from the system
> (`apt: python3-picamera2`).

---

## 4 — `app.py`  (the brain — line by line)

### 4.1 — Imports

```python
import atexit
import cv2
from flask import Flask, Response, render_template
from picamera2 import Picamera2
```

| Import | Plain-English meaning |
|---|---|
| `atexit` | Lets us register a "clean-up" function that runs automatically when the program exits. |
| `cv2` | OpenCV — we use exactly one feature: `cv2.imencode()` to turn a raw image into a `.jpg`. |
| `Flask, Response, render_template` | Flask building blocks — create a web server, send responses, render HTML files. |
| `Picamera2` | The official Python library that talks to the Arducam/Pi camera hardware. |

---

### 4.2 — Configuration

```python
JPEG_QUALITY = 80
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
```

These are just settings written in ALL_CAPS (a Python convention for
constants):

- **JPEG_QUALITY 80** — how sharp each frame is (1 = potato quality,
  100 = pristine). 80 is a good balance between image quality and file
  size.
- **1280 × 720** — the resolution we ask the camera for (720p).

---

### 4.3 — Camera Initialisation

```python
picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"format": "BGR888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
)
picam2.configure(config)
picam2.start()
```

Step by step:

1. **`Picamera2()`** — creates a camera object. Think of it as *"hey
   camera, I want to talk to you"*.
2. **`create_video_configuration(...)`** — tells the camera how we want
   the images delivered:
   - `"BGR888"` — each pixel is three numbers: **B**lue, **G**reen,
     **R**ed (one byte each = 8+8+8 = "888"). This is the colour
     format OpenCV expects, so no conversion step is needed later.
   - `"size": (1280, 720)` — we want 720p frames.
3. **`picam2.configure(config)`** — applies that configuration.
4. **`picam2.start()`** — starts the camera streaming internally. Frames
   are now continuously buffered in memory, ready for us to grab.

---

### 4.4 — Camera Cleanup (the "exit handler")

```python
def _release_camera() -> None:
    try:
        picam2.stop()
        picam2.close()
        print("[camera] released")
    except Exception:
        pass

atexit.register(_release_camera)
```

**Why this matters:** The camera is a physical device. Only **one**
program can use it at a time. If our script crashes without telling the
camera "I'm done", the camera stays locked and you'd have to reboot
the Pi. This code guarantees the camera is freed:

- `picam2.stop()` — stops the internal stream.
- `picam2.close()` — releases the hardware entirely.
- `atexit.register(...)` — tells Python: *"whenever this program ends —
  normally, via Ctrl-C, or in a crash — run `_release_camera()` first"*.

---

### 4.5 — `process_frame()` (the YOLO placeholder)

```python
def process_frame(frame):
    # Future: YOLO inference goes here
    return frame
```

This is deliberately a **do-nothing** function right now. It takes a
frame in and hands the **exact same frame** back out.

**Why does it exist?** Architecture. When you're ready to add YOLO
object detection, you'll put your AI code *inside this function*. The
rest of the app doesn't need to change at all — it just keeps calling
`process_frame(frame)` and streaming whatever comes back. This is called
the **separation of concerns** pattern.

Future version might look like:

```python
def process_frame(frame):
    results = model(frame)           # run AI detection
    frame = results[0].plot()        # draw bounding boxes on the frame
    return frame
```

---

### 4.6 — `generate_frames()` (the heart of the stream)

```python
def generate_frames():
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    while True:
        frame = picam2.capture_array()       # 1) grab a frame
        frame = process_frame(frame)         # 2) process it (no-op for now)
        ok, buffer = cv2.imencode(           # 3) compress to JPEG
            ".jpg", frame, encode_params
        )
        if not ok:
            continue

        yield (                              # 4) send it out
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )
```

This is a **generator** — a special Python function that uses `yield`
instead of `return`. Instead of running once and giving back one answer,
it runs **forever** in a loop, handing out one JPEG at a time whenever
the browser asks for the next one.

Let's walk through each step:

| Step | Code | What happens |
|---|---|---|
| 1 | `picam2.capture_array()` | Grabs the latest frame from the camera as a raw numpy array (a big grid of pixel values). |
| 2 | `process_frame(frame)` | Passes through the processing pipeline (currently does nothing). |
| 3 | `cv2.imencode(".jpg", ...)` | Compresses the raw pixel grid into JPEG format (much smaller). |
| 4 | `yield b"--frame\r\n..."` | Hands the JPEG bytes to Flask, wrapped in a special HTTP envelope (explained below). |

#### What's that weird `b"--frame\r\n..."` stuff?

That's the **MJPEG protocol**. The browser and server agree on a
"boundary" marker — the text `--frame`. Every time the browser sees
that marker, it knows: *"everything after this until the next marker is
a brand new JPEG image — throw away the old one and display this one
instead"*.

```
--frame                          ← boundary marker
Content-Type: image/jpeg         ← "the next blob is a JPEG"
                                 ← blank line = "headers are done"
<...thousands of JPEG bytes...>  ← the actual image data
--frame                          ← next boundary, next image…
```

This repeats forever, and the browser swaps images so fast it looks like
video.

---

### 4.7 — Flask Routes (the web server)

```python
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
```

Flask is the web server. A **route** is a URL pattern:

| URL | What the browser gets |
|---|---|
| `/` (homepage) | The `index.html` page (which contains the live-stream `<img>` tag) |
| `/video_feed` | The never-ending MJPEG stream from `generate_frames()` |

The special mimetype `multipart/x-mixed-replace; boundary=frame` tells
the browser: *"I'm going to keep sending you parts separated by the
word `frame` — every new part replaces the last one."*

---

### 4.8 — Entry Point

```python
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        _release_camera()
```

| Setting | Meaning |
|---|---|
| `host="0.0.0.0"` | Listen on all network interfaces — not just localhost. This is what lets your phone see the stream from across the room. |
| `port=5000` | The "door number" on the Pi. You'll type `http://<pi-ip>:5000` in your browser. |
| `threaded=True` | Let multiple browsers connect at once (each gets its own thread). |
| `try/finally` | Even if the server crashes, **always** release the camera. |

---

## 5 — `templates/index.html`  (the web page)

The important part is one line:

```html
<img id="live-stream" src="/video_feed" alt="Live camera stream" />
```

That `src="/video_feed"` tells the browser to make a request to our
`/video_feed` route. The browser receives the MJPEG stream and
continuously replaces the image — producing live video.

The rest of the HTML is purely cosmetic:
- Dark background (`#0b0f19`)
- A gradient title ("YOLO-Fun — Live Camera Feed")
- A pulsing red "Live" badge overlaid on the video
- A footer line

---

## 6 — How To Run It

**On the Raspberry Pi:**

```bash
# 1. Install Python dependencies (one time)
pip install -r requirements.txt

# 2. Start the server
python app.py
```

You should see:

```
[camera] Picamera2 started  resolution=1280×720  format=BGR888
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
```

**On your phone / laptop:**

1. Make sure you're on the **same Wi-Fi** as the Pi.
2. Find the Pi's IP address (run `hostname -I` on the Pi).
3. Open a browser and go to `http://<that-ip>:5000`.
4. You should see live video.

To stop the server: press **Ctrl + C** in the terminal. The camera will
be released automatically.

---

## 7 — Glossary

| Term | Meaning |
|---|---|
| **CSI** | Camera Serial Interface — the ribbon cable connecting the Arducam to the Pi's camera port. |
| **Flask** | A lightweight Python library for building web servers. |
| **Frame** | A single still image captured by the camera. Many frames per second = video. |
| **Generator** | A Python function that uses `yield` to produce values one at a time in a loop, instead of computing everything at once. |
| **ISP** | Image Signal Processor — hardware on the Pi that converts raw sensor data into usable images. |
| **JPEG** | A compressed image format. Smaller than raw pixels, fast to encode/decode. |
| **libcamera** | The low-level Linux camera framework that Picamera2 talks to under the hood. |
| **MJPEG** | Motion JPEG — a video streaming method that sends a sequence of JPEG images over HTTP. |
| **NumPy array** | A grid of numbers in memory. An image is a 3D array: height × width × 3 colour channels. |
| **OpenCV (cv2)** | A computer-vision library. We use it only for JPEG compression here. |
| **Picamera2** | The official Python library for Raspberry Pi cameras. Replaces the older `picamera` library. |
| **Route** | A URL pattern (like `/` or `/video_feed`) that the web server knows how to respond to. |
| **V4L2** | Video4Linux2 — a Linux interface for cameras. We bypass it because it has bugs with Picamera2 on Bookworm. |
| **YOLO** | "You Only Look Once" — a fast AI model family for detecting objects in images. Future addition to this project. |

---

## 8 — What's Next?

When you're ready to add YOLO object detection, the **only file you
need to touch** is `app.py`, and the **only function you need to edit**
is `process_frame()`. The rest of the streaming pipeline stays exactly
the same.
