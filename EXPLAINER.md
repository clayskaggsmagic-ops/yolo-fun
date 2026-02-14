# YOLO-Fun — How Everything Works (Deep Dive)

A complete beginner's guide to every line of code in this project.
No prior Python, web, or computer-vision knowledge assumed.

---

## Table of Contents

1. [The Big Picture](#1--the-big-picture)
2. [The Files](#2--the-files)
3. [Why Not PyTorch / Ultralytics?](#3--why-not-pytorch--ultralytics)
4. [requirements.txt](#4--requirementstxt-the-shopping-list)
5. [app.py — Imports](#5--apppy--imports)
6. [app.py — Configuration Constants](#6--apppy--configuration-constants)
7. [app.py — Camera Initialisation](#7--apppy--camera-initialisation)
8. [app.py — YOLO Model Loading (cv2.dnn)](#8--apppy--yolo-model-loading-cv2dnn)
9. [app.py — Camera Cleanup (atexit)](#9--apppy--camera-cleanup-atexit)
10. [app.py — Scale Factors](#10--apppy--scale-factors)
11. [app.py — process_frame() — The Brain](#11--apppy--process_frame--the-brain)
    - [Step 1: Colour Fix](#step-1--colour-fix)
    - [Step 2: Blob Creation](#step-2--blob-creation)
    - [Step 3: Forward Pass](#step-3--forward-pass)
    - [Step 4: Tensor Parsing](#step-4--tensor-parsing)
    - [Step 5–6: Filtering + NMS](#steps-56--filtering--nms)
    - [Step 7: Annotation](#step-7--annotation)
12. [app.py — generate_frames()](#12--apppy--generate_frames-the-heartbeat)
13. [app.py — Flask Routes](#13--apppy--flask-routes-the-web-server)
14. [app.py — Entry Point](#14--apppy--entry-point)
15. [templates/index.html](#15--templatesindexhtml-the-web-page)
16. [How All the Pieces Connect](#16--how-all-the-pieces-connect)
17. [How to Run It](#17--how-to-run-it)
18. [Glossary](#18--glossary)

---

## 1 — The Big Picture

You have a **Raspberry Pi 4B** with a **camera** attached via a ribbon
cable. You want to:

1. See what the camera sees **live** on any phone or laptop on your
   Wi-Fi.
2. Have an **AI model** detect people in the frame and draw green boxes
   around them — in real time.

The data flows like this:

```
┌──────────────┐         ┌──────────────────────────┐         ┌──────────────┐
│   Camera     │  ──▶    │   Python app on the Pi    │  ──▶    │  Your phone  │
│  (Arducam    │  raw    │  1. Grab frame            │  JPEG   │  or laptop   │
│   IMX708)    │  pixels │  2. Fix colours            │  stream │  (browser)   │
│              │         │  3. Build 640×640 blob     │         │              │
│              │         │  4. YOLO neural-net pass   │         │              │
│              │         │  5. Filter: persons only   │         │              │
│              │         │  6. Remove duplicate boxes │         │              │
│              │         │  7. Draw green boxes       │         │              │
│              │         │  8. Compress to JPEG       │         │              │
│              │         │  9. Stream over HTTP       │         │              │
└──────────────┘         └──────────────────────────┘         └──────────────┘
```

The streaming trick is called **MJPEG** (Motion JPEG). The server
sends a never-ending sequence of JPEG images; the browser replaces the
displayed image each time a new one arrives, making it look like video.

---

## 2 — The Files

```
yolo-fun/
├── app.py                    ← The main Python program (server + AI)
├── yolo11n.onnx              ← The AI model file (you create this once)
├── requirements.txt          ← Lists the Python packages we need
└── templates/
    └── index.html            ← The web page your browser loads
```

---

## 3 — Why Not PyTorch / Ultralytics?

You might see YOLO tutorials that say `from ultralytics import YOLO`.
That pulls in **PyTorch**, a huge machine-learning framework. Here's
why we can't use it:

| Problem | Detail |
|---|---|
| **CPU mismatch** | PyTorch ≥ 2.6 is compiled with ARMv8.2 instructions (optimised for Pi 5). The Pi 4 has an ARMv8.0 CPU. When the Pi 4 hits an instruction it physically doesn't have, the Linux kernel kills the program with **SIGILL** (Illegal Instruction) |
| **No downgrade path** | PyTorch 2.5.1 supports ARMv8.0, but has no pre-built wheel for Python 3.13. Building from source takes hours and often fails |
| **System Python is locked** | Downgrading to Python 3.11 would break the system-installed `picamera2` driver |

**The escape route:** OpenCV has its own inference engine (`cv2.dnn`)
that can load ONNX models directly. Its C++ backend **probes the CPU
at runtime** and only uses instructions the hardware actually supports.
No PyTorch needed. No SIGILL.

---

## 4 — `requirements.txt` (the shopping list)

```text
flask>=3.0,<4.0
opencv-python-headless>=4.8,<5.0
numpy>=1.26,<3.0
```

You install these with `pip install -r requirements.txt`. Python
downloads three libraries:

| Package | What it does |
|---|---|
| **Flask** | Lightweight web-server framework — lets us say *"when someone visits this URL, give them this response"* |
| **opencv-python-headless** | Computer-vision library — we use it for: colour conversion, blob creation, neural-net inference (`cv2.dnn`), JPEG encoding, and drawing boxes. The `-headless` suffix skips GUI code the Pi doesn't need |
| **NumPy** | Numerical array library — lets us slice and reshape the raw tensor output from the neural network |

### What about Picamera2?

Pre-installed on Raspberry Pi OS. Do **not** pip-install it. It comes
from the system APT package manager (`apt install python3-picamera2`).

### Version range syntax

`>=3.0,<4.0` means *"any version from 3.0 up to (but not including)
4.0"*. This ensures compatibility without locking to one exact number.

---

## 5 — `app.py` — Imports

```python
import atexit                            # Line 23

import cv2                               # Line 25
import numpy as np                       # Line 26
from flask import Flask, Response, render_template   # Line 27
from picamera2 import Picamera2          # Line 28
```

### What is `import`?

Python code is organized into **modules** (files) and **packages**
(folders of files). `import` means: *"Go find this other file and make
its tools available to me."*

### Each import explained

| Import | What it gives us | Where it comes from |
|---|---|---|
| `atexit` | Register "run this function when the program exits" | Python's standard library (always available) |
| `cv2` | OpenCV's Python interface — image manipulation, neural-net inference, drawing, encoding | The `opencv-python-headless` pip package |
| `numpy as np` | Fast array math — we use it to reshape the model's output tensor and find the highest class score | The `numpy` pip package. `as np` is an **alias** — lets us type `np.argmax()` instead of `numpy.argmax()` |
| `Flask` | A class that *is* the web server | The `flask` pip package |
| `Response` | A class for building HTTP responses — we use it for the streaming reply | Same `flask` package |
| `render_template` | Reads an HTML file from `templates/` and returns it as a string | Same `flask` package |
| `Picamera2` | A class that controls the physical camera hardware | Pre-installed system package on the Pi |

### `from X import Y` vs `import X`

- `import cv2` — imports the whole module; access its contents as
  `cv2.something`.
- `from flask import Flask` — reaches *into* the `flask` module and
  grabs just the `Flask` name, so you can write `Flask(...)` directly
  instead of `flask.Flask(...)`.

### `import numpy as np`

`as np` creates a shortcut name. `numpy` is 5 characters; `np` is 2.
Since we type it a lot, the alias saves keystrokes. This is a near‑
universal convention — almost every Python codebase calls NumPy `np`.

---

## 6 — `app.py` — Configuration Constants

```python
JPEG_QUALITY      = 80        # Line 33
FRAME_WIDTH       = 1280      # Line 34
FRAME_HEIGHT      = 720       # Line 35
MODEL_INPUT_SIZE  = 640       # Line 36
CONFIDENCE_THRESH = 0.50      # Line 37
NMS_THRESH        = 0.40      # Line 38
PERSON_CLASS_ID   = 0         # Line 39
```

### Why ALL_CAPS?

A Python **convention** (not enforced by the language) meaning: *"This
is a constant — set it once at the top of the file and don't change it
while the program runs."*

| Constant | Value | Meaning |
|---|---|---|
| `JPEG_QUALITY` | `80` | JPEG compression quality (1–100). 80 = good balance of sharpness vs. file size |
| `FRAME_WIDTH` | `1280` | Camera capture width (720p HD) |
| `FRAME_HEIGHT` | `720` | Camera capture height |
| `MODEL_INPUT_SIZE` | `640` | YOLO models are trained on 640×640 images. We must resize every frame to this size before feeding it to the network |
| `CONFIDENCE_THRESH` | `0.50` | Only draw a box if the AI is ≥ 50% confident. Prevents jittery false positives |
| `NMS_THRESH` | `0.40` | Overlap threshold for Non-Maximum Suppression (explained later). If two boxes overlap by more than 40%, keep only the stronger one |
| `PERSON_CLASS_ID` | `0` | In the COCO dataset YOLO was trained on, class 0 = "person". There are 80 classes total (car, dog, chair…) — we only care about people |

---

## 7 — `app.py` — Camera Initialisation

```python
picam2 = Picamera2()                                          # Line 44

config = picam2.create_video_configuration(                   # Line 46
    main={"format": "BGR888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
)
picam2.configure(config)                                      # Line 49
picam2.start()                                                # Line 50
```

### `picam2 = Picamera2()`

Creates a **camera object** — a chunk of code that talks to the Arducam
hardware. The trailing `()` calls the **constructor** (the special
function that creates a new instance). Think of it as pressing the "on"
button.

### `create_video_configuration(...)`

Builds a settings dictionary:

```python
main={"format": "BGR888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
```

- **`main={...}`** — configures the primary output stream.
- **`"format": "BGR888"`** — each pixel is three bytes: **B**lue,
  **G**reen, **R**ed. This is the colour order OpenCV expects.
  "888" = 8 bits per channel = 256 brightness levels per colour.
- **`"size": (1280, 720)`** — the resolution as a **tuple**
  (width, height).

### `picam2.configure(config)` / `picam2.start()`

`.configure()` applies the settings to the hardware driver.
`.start()` begins continuous capture — the camera's ISP now constantly
fills a buffer in memory with fresh frames.

### The `print(f"...")` line

```python
print(f"[camera] Picamera2 started  "
      f"resolution={FRAME_WIDTH}×{FRAME_HEIGHT}  format=BGR888")
```

Prints a startup message. The `f"..."` is an **f-string** — anything
inside `{braces}` gets replaced with the variable's value. So
`{FRAME_WIDTH}` becomes `1280`.

---

## 8 — `app.py` — YOLO Model Loading (cv2.dnn)

```python
MODEL_PATH = "yolo11n.onnx"                   # Line 58

net = cv2.dnn.readNetFromONNX(MODEL_PATH)      # Line 60
```

### What is ONNX?

**ONNX** (Open Neural Network Exchange) is a universal file format for
AI models. You can export a model from any framework (PyTorch,
TensorFlow, etc.) into a `.onnx` file, and any compatible runtime can
load and run it. Think of it like a `.pdf` — one file format, many
readers.

### What is `cv2.dnn`?

OpenCV has a built-in **DNN (Deep Neural Network)** module that can
load and execute neural network models without needing PyTorch or
TensorFlow. Its C++ backend checks what CPU instructions are available
at runtime, so it won't crash on older hardware like the Pi 4.

### `net = cv2.dnn.readNetFromONNX(MODEL_PATH)`

This reads the entire YOLO11n model from the `.onnx` file and loads it
into memory. After this line, `net` is a neural-network object we can
feed images into and get detection results out.

We do this **once at startup** because loading takes seconds and uses
significant memory. If we reloaded it per-frame, the Pi would crash.

### Why is this outside any function?

Code at the "top level" of a Python file (not indented inside a `def`
or `class`) runs **once** when the file is first executed. That's
exactly what we want.

---

## 9 — `app.py` — Camera Cleanup (`atexit`)

```python
def _release_camera() -> None:                 # Line 68
    try:
        picam2.stop()
        picam2.close()
        print("[camera] released")
    except Exception:
        pass

atexit.register(_release_camera)               # Line 78
```

### Why we need this

The camera is a **physical device** — only one program can use it at a
time. If our program crashes without releasing it, the Pi's kernel
thinks the camera is busy. You'd have to reboot. This cleanup prevents
that.

### Syntax breakdown

| Code | Meaning |
|---|---|
| `def _release_camera()` | Defines a function. Leading `_` = convention for "private/internal" |
| `-> None` | **Type hint**: this function returns nothing. Documentation only — doesn't change behaviour |
| `try:` | "Try executing this code…" |
| `except Exception:` | "…but if anything goes wrong, catch the error and…" |
| `pass` | "…do nothing." (We're shutting down anyway, so no useful action to take) |
| `picam2.stop()` | Stops the ISP from producing frames |
| `picam2.close()` | Releases the hardware device entirely |
| `atexit.register(...)` | "When Python exits (for any reason), call this function first" |

### Why no parentheses?

```python
atexit.register(_release_camera)     # ← no ()
```

We pass the function **itself** as a reference, not the result of
calling it. `atexit` stores the reference and calls it later at exit.

| Syntax | Meaning |
|---|---|
| `_release_camera()` | **Call** the function right now |
| `_release_camera` | **Reference** the function object (don't call it) |

---

## 10 — `app.py` — Scale Factors

```python
_sx = FRAME_WIDTH  / MODEL_INPUT_SIZE   # 1280 / 640 = 2.0
_sy = FRAME_HEIGHT / MODEL_INPUT_SIZE   # 720  / 640 = 1.125
```

### The problem these solve

The YOLO model sees a 640×640 image. But our camera produces 1280×720
frames. When the model says "there's a person at x=200 in the 640×640
image", we need to translate that back to x=400 in the 1280-pixel-wide
real frame.

`_sx = 2.0` means "multiply every horizontal coordinate by 2".
`_sy = 1.125` means "multiply every vertical coordinate by 1.125".

### Why precompute?

These values never change — the resolution is fixed. Computing them
once outside the function (rather than 8,400 times per frame inside
the loop) is a tiny speed optimization.

### Why the leading underscore?

Same convention as `_release_camera`: the `_` prefix signals "this
is an internal/private variable, not meant to be part of the public
API".

---

## 11 — `app.py` — `process_frame()` — The Brain

This is the most complex function. Let's go through every step.

### Function signature

```python
def process_frame(frame):
```

Takes one argument: `frame`, a **NumPy array** — a giant 3D grid of
numbers representing pixel colours. Shape: `(720, 1280, 3)`:
- 720 rows (height)
- 1280 columns (width)
- 3 colour values per pixel (0–255 each)

---

### Step 1 — Colour Fix

```python
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
```

**The problem:** Even though we asked for `BGR888`, Picamera2 actually
delivers pixels in **RGB** order. OpenCV expects **BGR**. Without this
swap, reds and blues flip — you look like a Smurf.

`cv2.cvtColor()` = "convert colour". `cv2.COLOR_RGB2BGR` = "swap the
Red and Blue channels". It modifies every pixel in the array: pixel
`[R, G, B]` becomes `[B, G, R]`.

---

### Step 2 — Blob Creation

```python
blob = cv2.dnn.blobFromImage(
    frame, 1 / 255.0, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
    swapRB=False, crop=False,
)
```

Neural networks can't eat raw images. They need a specific input
format called a **blob**. `blobFromImage` does three things at once:

| Argument | What it does |
|---|---|
| `frame` | The input image (1280×720, BGR, pixels from 0–255) |
| `1 / 255.0` | **Scale factor** — divides every pixel value by 255, normalising them to the 0.0–1.0 range. Neural networks work better with small numbers |
| `(640, 640)` | **Resize** the image to 640×640 pixels (what YOLO expects) |
| `swapRB=False` | Don't swap Red/Blue — we already fixed the channels in step 1 |
| `crop=False` | Stretch the image to fit 640×640, don't crop the edges |

The output `blob` is a 4D array with shape `(1, 3, 640, 640)`:
- `1` — batch size (one image)
- `3` — colour channels (B, G, R)
- `640, 640` — width and height

This shape is what all modern neural-net frameworks expect: **NCHW**
(batch-**N**, **C**hannels, **H**eight, **W**idth).

---

### Step 3 — Forward Pass

```python
net.setInput(blob)
outputs = net.forward()          # shape: [1, 84, 8400]
```

**`net.setInput(blob)`** — loads the blob into the network's input slot.

**`net.forward()`** — runs the **forward pass**: pushes the input
through all ~80 layers of the neural network (convolutions, activations,
pooling, etc.) and collects the output. This is where the actual "AI
thinking" happens. On a Pi 4 this takes roughly 200–500ms.

The output shape is `[1, 84, 8400]`:
- `1` — batch size (one image)
- `84` — 4 bounding-box values + 80 class scores
- `8400` — the number of candidate detections the model generates
  (grid cells at multiple scales)

---

### Step 4 — Tensor Parsing

```python
preds = outputs[0].T             # shape: [8400, 84]
```

The raw output is awkwardly shaped for iteration. We fix it:

1. **`outputs[0]`** — strips the batch dimension: `[1, 84, 8400]` →
   `[84, 8400]`
2. **`.T`** — **transpose** (flip rows and columns): `[84, 8400]` →
   `[8400, 84]`

Now each **row** is one candidate detection, and each row looks like:

```
index:  0          1          2       3       4       5       ... 83
value:  x_center   y_center   width   height  person  bicycle ... toothbrush
        ────────── ────────── ─────── ─────── ─────── ─────── ─── ──────────
        bounding box (640-space)      ← 80 class confidence scores →
```

There are 8,400 rows — most are garbage (low confidence). The next
step filters them.

---

### Steps 5–6 — Filtering + NMS

```python
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
```

This is dense. Let's unpack every line:

#### `boxes = []` and `confidences = []`

Create two empty **lists**. We'll fill them with valid detections as we
loop.

#### `for row in preds:`

Loop through all 8,400 candidate detections.

#### `class_scores = row[4:]`

**Array slicing**: `row[4:]` means "give me everything from index 4
onward". Indices 0–3 are the bounding box; indices 4–83 are the 80
class scores. So `class_scores` is an array of 80 numbers.

#### `class_id = int(np.argmax(class_scores))`

`np.argmax()` finds the **index** of the highest value in the array.
If the "person" score is the highest, this returns `0`. If "dog" is
highest, it returns `16`. We wrap it in `int()` to convert from a NumPy
integer to a plain Python integer.

#### `conf = float(class_scores[class_id])`

Looks up the actual score at that index. For example, if `class_id` is
`0` and `class_scores[0]` is `0.87`, then `conf = 0.87` (87% confident
it's a person).

#### The filter

```python
if class_id != PERSON_CLASS_ID or conf < CONFIDENCE_THRESH:
    continue
```

`continue` = *"skip the rest of this loop iteration, move to the next
row"*. We skip if:
- The top class isn't "person" (`!= 0`), **or**
- The confidence is below 50% (`< 0.50`)

This eliminates the vast majority of the 8,400 rows.

#### Coordinate conversion

```python
cx, cy, w, h = row[0], row[1], row[2], row[3]
x = int((cx - w / 2) * _sx)
y = int((cy - h / 2) * _sy)
w = int(w * _sx)
h = int(h * _sy)
```

The model outputs boxes as **(centre-x, centre-y, width, height)** in
640×640 space. We need **(top-left-x, top-left-y, width, height)** in
1280×720 space:

```
640×640 model space              1280×720 real frame
┌────────────────────┐           ┌─────────────────────────────────┐
│          (cx,cy)   │           │                                 │
│         ○          │    ×_sx   │         (x,y)──────────┐        │
│      ┌──┼──┐       │   ×_sy   │           │            │        │
│      │  │  │       │  ──────▶ │           │   PERSON   │        │
│      └──┴──┘       │           │           │            │        │
│        w,h         │           │           └────────────┘        │
└────────────────────┘           │             w*_sx, h*_sy        │
                                 └─────────────────────────────────┘
```

- `cx - w/2` converts centre-x to **left edge** x.
- `cy - h/2` converts centre-y to **top edge** y.
- Multiplying by `_sx` and `_sy` scales from 640-space to native space.
- `int(...)` rounds to whole pixels.

#### `boxes.append(...)` / `confidences.append(...)`

`.append()` adds an item to the end of a list.

#### Non-Maximum Suppression (NMS)

```python
indices = cv2.dnn.NMSBoxes(boxes, confidences,
                           CONFIDENCE_THRESH, NMS_THRESH)
```

**The problem NMS solves:** YOLO often generates multiple overlapping
boxes for the same person — e.g. five boxes all surrounding your face,
each from a different grid cell. NMS keeps only the **strongest** one
and eliminates boxes that overlap it by more than `NMS_THRESH` (40%).

Visually:

```
Before NMS:                    After NMS:
┌──────────┐                   ┌──────────┐
│┌────────┐│                   │          │
││┌──────┐││                   │  PERSON  │
│││PERSON │││   ──────▶        │   92%    │
││└──────┘││                   │          │
│└────────┘│                   └──────────┘
└──────────┘
  92%, 88%, 85%                  92% (winner)
```

`indices` is a list of which items in `boxes` survived the suppression.

---

### Step 7 — Annotation

```python
for i in indices:
    idx = int(i) if isinstance(i, (int, np.integer)) else int(i[0])
    x, y, w, h = boxes[idx]
    conf = confidences[idx]
```

#### The `isinstance` check

Different OpenCV versions return `indices` in different formats — either
a plain `int` or a 1-element array `[int]`. This line handles both:

```python
idx = int(i) if isinstance(i, (int, np.integer)) else int(i[0])
```

This is a **ternary expression** (Python's inline if/else):

```
value_if_true  if  condition  else  value_if_false
```

If `i` is already an integer, use it directly. Otherwise, grab the
first element with `i[0]`.

#### Drawing the rectangle

```python
cv2.rectangle(frame, (x, y), (x + w, y + h),
              color=(0, 255, 0), thickness=2)
```

| Argument | Meaning |
|---|---|
| `frame` | The image to draw on (modified in place) |
| `(x, y)` | Top-left corner of the box |
| `(x + w, y + h)` | Bottom-right corner (top-left + width/height) |
| `(0, 255, 0)` | Colour in BGR: B=0, G=255, R=0 = **bright green** |
| `thickness=2` | Line width in pixels |

#### Drawing the confidence label

```python
label = f"Person {conf:.0%}"
```

An f-string with a **format specifier**: `{conf:.0%}` means:
1. Take `conf` (e.g. `0.87`)
2. Multiply by 100 → `87`
3. Round to 0 decimal places → `87`
4. Append `%` → `87%`

Result: `"Person 87%"`

```python
(tw, th), baseline = cv2.getTextSize(
    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2,
)
```

Measures how big the text will be **before drawing it** (in pixels).
We need this to draw a background rectangle behind the text so it's
readable.

- `tw, th` = text width and height
- `baseline` = extra space below (unused)

```python
cv2.rectangle(frame, (x, y - th - 10), (x + tw, y),
              color=(0, 255, 0), thickness=-1)
```

Draws a **filled** green rectangle (`thickness=-1` = fill) just above
the bounding box. This is the label background.

```python
cv2.putText(frame, label, (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 0), 2)
```

Draws the text on top in **black** `(0, 0, 0)` so it contrasts with
the green background.

#### Return

```python
return frame
```

The frame now has green boxes and labels drawn on it. It goes back to
`generate_frames()` for JPEG compression.

---

## 12 — `app.py` — `generate_frames()` (the heartbeat)

```python
def generate_frames():
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    while True:
        frame = picam2.capture_array()
        frame = process_frame(frame)
        ok, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )
```

### What is a generator?

A normal function runs, returns one value, and is done. A **generator**
uses `yield` instead of `return`. Each time you ask it for the next
value:

1. It **resumes** from where it last `yield`ed
2. Runs until it hits another `yield`
3. **Pauses** and hands out that value

Since there's a `while True:` (infinite loop), this generator never
stops. Flask keeps asking for the next chunk, and it keeps yielding
JPEG frames.

### Line by line

#### `encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]`

A key-value pair in list form telling `cv2.imencode` to use quality 80.
Created once outside the loop to avoid rebuilding it every frame.

#### `frame = picam2.capture_array()`

Grabs the latest frame from the camera's buffer as a NumPy array. This
is a **blocking call** — if no new frame is ready, it waits.

#### `ok, buffer = cv2.imencode(".jpg", frame, encode_params)`

Compresses the raw pixel array into JPEG format:
- Input: `frame` — 720 × 1280 × 3 = ~2.7 MB of raw pixels
- Output: `buffer` — ~50–100 KB of JPEG data

Returns two values (**tuple unpacking**):
- `ok` — `True` if encoding succeeded
- `buffer` — the compressed bytes

#### The `yield` block

```python
yield (
    b"--frame\r\n"
    b"Content-Type: image/jpeg\r\n\r\n"
    + buffer.tobytes()
    + b"\r\n"
)
```

**`b"..."`** — a **bytes literal**. HTTP transmits raw binary, not
Unicode text.

**`\r\n`** — carriage return + line feed, the standard HTTP line ending.

**`buffer.tobytes()`** — converts the NumPy byte array into plain
Python `bytes` for concatenation.

The full chunk, laid out:

```
--frame\r\n                          ← "Here comes a new part"
Content-Type: image/jpeg\r\n\r\n     ← "It's a JPEG" + blank line
[...87,000 bytes of JPEG data...]    ← The actual image
\r\n                                 ← End of this part
```

The browser displays the JPEG, then waits for the next `--frame`.
When it arrives, the browser **replaces** the image. This happens
many times per second → live video.

---

## 13 — `app.py` — Flask Routes (the web server)

### Creating the app

```python
app = Flask(__name__)
```

`Flask(...)` creates a web-server application. `__name__` is a special
Python variable holding the current module's name — Flask uses it to
find the `templates/` folder.

### The `@app.route(...)` decorator

```python
@app.route("/")
def index():
    return render_template("index.html")
```

The `@` symbol is a **decorator** — special syntax that wraps a function
with extra behaviour. `@app.route("/")` tells Flask: *"When a browser
visits `/`, call this function and send back its return value."*

`render_template("index.html")` reads `templates/index.html` and
returns it as a string.

### The streaming route

```python
@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
```

When a browser requests `/video_feed`, Flask returns a `Response`
wrapping our generator:

- **`generate_frames()`** — the infinite generator. Flask calls `next()`
  on it repeatedly, sending each yielded chunk.
- **`mimetype="multipart/x-mixed-replace; boundary=frame"`** — tells
  the browser:
  - `multipart` — "Multiple parts coming"
  - `x-mixed-replace` — "Each part **replaces** the previous one"
  - `boundary=frame` — "Parts are separated by `--frame`"

This is the core trick that makes MJPEG work.

---

## 14 — `app.py` — Entry Point

```python
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        _release_camera()
```

### `if __name__ == "__main__":`

When you run `python app.py`, Python sets `__name__` to `"__main__"`.
If this file were imported by another file, `__name__` would be the
module name instead. This block says: *"Only start the server if
someone ran this file directly."*

### `app.run(...)`

| Argument | Meaning |
|---|---|
| `host="0.0.0.0"` | Listen on **all** network interfaces — lets other devices on the Wi-Fi reach the stream |
| `port=5000` | The port number. Browsers connect to `http://<pi-ip>:5000` |
| `threaded=True` | Each request gets its own **thread**, so multiple browsers can watch at once |

### `try: ... finally:`

`finally` is the **guarantee block** — no matter how `app.run()` ends
(Ctrl-C, crash, exception), the camera is always released. This is the
"belt-and-suspenders" approach: `atexit` is the belt, `finally` is the
suspenders.

---

## 15 — `templates/index.html` (the web page)

The only functionally important line:

```html
<img id="live-stream" src="/video_feed" alt="Live camera stream" />
```

| Attribute | What it does |
|---|---|
| `src="/video_feed"` | Makes a request to our MJPEG route. Because the response is `multipart/x-mixed-replace`, the browser continuously swaps in new images |
| `alt="..."` | Fallback text if the image can't load |

Everything else is cosmetic: dark background, gradient title, pulsing
red "Live" badge, responsive sizing.

---

## 16 — How All the Pieces Connect

Full data flow — every step, every time a frame is streamed:

```
 ①  Camera sensor captures photons → raw Bayer data
                    │
 ②  Picamera2 ISP converts Bayer → RGB pixel array (numpy)
                    │
 ③  generate_frames() calls picam2.capture_array()
                    │
 ④  generate_frames() calls process_frame(frame)
         │
         ├─ ⑤  cv2.cvtColor: swap RGB → BGR
         │
         ├─ ⑥  cv2.dnn.blobFromImage: resize 1280×720 → 640×640, normalise 0–1
         │
         ├─ ⑦  net.setInput + net.forward: run 80-layer neural network
         │       → raw tensor [1, 84, 8400]
         │
         ├─ ⑧  Transpose → [8400, 84], loop rows, extract class + confidence
         │
         ├─ ⑨  Filter: keep only class 0 (person) with conf > 50%
         │
         ├─ ⑩  NMSBoxes: remove overlapping duplicate boxes
         │
         ├─ ⑪  Scale 640→1280/720, draw green boxes + "Person 87%" labels
         │
         └─ ⑫  return annotated BGR frame
                    │
 ⑬  cv2.imencode: compress BGR frame → JPEG bytes (~50–100 KB)
                    │
 ⑭  yield: wrap JPEG in MJPEG multipart boundary
                    │
 ⑮  Flask sends bytes over HTTP to the browser
                    │
 ⑯  Browser <img> tag receives JPEG, displays it, waits for next one
                    │
          [loop back to ① — runs continuously]
```

### Expected performance on Raspberry Pi 4B

| Step | Approximate time |
|---|---|
| Camera capture | ~30 ms |
| RGB→BGR | ~2 ms |
| Blob creation + resize | ~5 ms |
| YOLO forward pass (cv2.dnn, 4-core ARM) | ~200–500 ms |
| Tensor parsing + NMS | ~5 ms |
| Drawing + JPEG encoding | ~5 ms |
| **Total per frame** | **~250–550 ms → about 2–4 FPS** |

2–4 FPS is typical for YOLO on a Pi 4 without a GPU or NPU. It's not
silky smooth, but it updates fast enough to be useful for person
detection and alerting.

---

## 17 — How to Run It

### First-time setup (on the Pi)

```bash
# 1. Clone the repo
git clone https://github.com/clayskaggsmagic-ops/yolo-fun.git
cd yolo-fun

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Get the YOLO model in ONNX format
#    Option A: export on a machine that HAS ultralytics:
#      python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx')"
#      scp yolo11n.onnx pi@<pi-ip>:~/yolo-fun/
#
#    Option B: download a pre-exported yolo11n.onnx from the Ultralytics releases
```

### Running the server

```bash
python app.py
```

You should see:

```
[camera] Picamera2 started  resolution=1280×720  format=BGR888
[yolo]   loaded yolo11n.onnx  (cv2.dnn / ONNX)
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
```

### Viewing the stream

1. Find the Pi's IP: `hostname -I` (e.g. `192.168.1.42`)
2. On any device on the same Wi-Fi, open a browser
3. Go to `http://192.168.1.42:5000`
4. You should see live video with green boxes around people

### Stopping

Press **Ctrl + C**. The camera releases automatically.

---

## 18 — Glossary

| Term | Meaning |
|---|---|
| **atexit** | Python module — register functions to run when the program exits |
| **BGR** | Blue-Green-Red — OpenCV's colour channel order (opposite of RGB) |
| **Blob** | A preprocessed 4D array (batch, channels, height, width) ready for neural-net input |
| **Bounding box** | A rectangle drawn around a detected object |
| **Bytes literal** | `b"hello"` — raw binary data, not Unicode text |
| **Class ID** | Number identifying what YOLO detected (0 = person, 1 = bicycle, etc.) |
| **COCO** | Common Objects in Context — the 80-class dataset YOLO was trained on |
| **Confidence** | 0.0–1.0 score: how sure the AI is about a detection |
| **CSI** | Camera Serial Interface — the ribbon cable connecting camera to Pi |
| **cv2.dnn** | OpenCV's built-in neural-network inference engine (no PyTorch needed) |
| **Decorator** | `@something` syntax that wraps a function with extra behaviour |
| **f-string** | `f"text {variable}"` — Python string with embedded variable values |
| **Flask** | Lightweight Python web-server framework |
| **Forward pass** | Running input data through all layers of a neural network to get output |
| **Frame** | A single still image from the camera |
| **Generator** | A function using `yield` to produce values one at a time in a loop |
| **IoU** | Intersection over Union — measures how much two boxes overlap (used in NMS) |
| **ISP** | Image Signal Processor — hardware converting raw sensor data to usable pixels |
| **JPEG** | Compressed image format. Much smaller than raw pixel data |
| **libcamera** | Linux camera framework that Picamera2 uses internally |
| **MJPEG** | Motion JPEG — streaming video as a sequence of JPEGs over HTTP |
| **NCHW** | Tensor dimension order: batch-N, Channels, Height, Width |
| **NMS** | Non-Maximum Suppression — algorithm to remove duplicate overlapping detections |
| **NumPy** | Python library for fast numerical array operations |
| **ONNX** | Open Neural Network Exchange — universal model file format |
| **OpenCV (cv2)** | Computer-vision library for image processing and neural-net inference |
| **Picamera2** | Official Python library for Raspberry Pi cameras |
| **Route** | A URL pattern (`/`, `/video_feed`) mapped to a Python function |
| **SIGILL** | Signal: Illegal Instruction — kernel kills a program that tries to run a CPU instruction the hardware doesn't support |
| **Tensor** | A multi-dimensional array of numbers (the data format neural networks work with) |
| **Thread** | Lightweight parallel execution path — lets multiple requests run at once |
| **Transpose** | Flip rows and columns of a matrix (`.T`) |
| **Tuple unpacking** | `a, b = (1, 2)` — assign multiple variables from a collection |
| **Type hint** | `-> None` — annotation describing expected types (documentation only) |
| **V4L2** | Video4Linux2 — Linux kernel camera interface (bypassed in our setup) |
| **YOLO** | "You Only Look Once" — family of fast object-detection AI models |
