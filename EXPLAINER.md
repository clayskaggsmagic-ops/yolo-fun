# YOLO-Fun — How Everything Works (Deep Dive)

A complete beginner's guide to every line of code in this project.
No prior Python, web, or computer-vision knowledge assumed.

---

## Table of Contents

1. [The Big Picture](#1--the-big-picture)
2. [The Files](#2--the-files)
3. [requirements.txt](#3--requirementstxt-the-shopping-list)
4. [app.py — Imports](#4--apppy--imports)
5. [app.py — Configuration Constants](#5--apppy--configuration-constants)
6. [app.py — Camera Initialisation](#6--apppy--camera-initialisation)
7. [app.py — YOLO Model Loading](#7--apppy--yolo-model-loading)
8. [app.py — Camera Cleanup](#8--apppy--camera-cleanup-atexit)
9. [app.py — process_frame()](#9--apppy--process_frame-the-brain)
10. [app.py — generate_frames()](#10--apppy--generate_frames-the-heartbeat)
11. [app.py — Flask Routes](#11--apppy--flask-routes-the-web-server)
12. [app.py — Entry Point](#12--apppy--entry-point-if-__name__--__main__)
13. [templates/index.html](#13--templatesindexhtml-the-web-page)
14. [How All the Pieces Connect](#14--how-all-the-pieces-connect)
15. [How to Run It](#15--how-to-run-it)
16. [Glossary](#16--glossary)

---

## 1 — The Big Picture

You have a **Raspberry Pi** with a **camera** attached via a ribbon
cable. You want to see what the camera sees — live — on **any phone or
laptop** on your Wi-Fi, *and* have an AI model detect people in the
frame and draw green boxes around them.

```
┌──────────────┐         ┌─────────────────────────┐         ┌──────────────┐
│   Camera     │  ──▶    │   Python app on the Pi   │  ──▶    │  Your phone  │
│  (Arducam    │  raw    │  1. Grab frame           │  JPEG   │  or laptop   │
│   IMX708)    │  pixels │  2. Fix colours           │  stream │  (browser)   │
│              │         │  3. YOLO person detection │         │              │
│              │         │  4. Compress to JPEG      │         │              │
│              │         │  5. Stream over HTTP      │         │              │
└──────────────┘         └─────────────────────────┘         └──────────────┘
```

This streaming trick is called **MJPEG** (Motion JPEG). The server
sends a never-ending sequence of JPEG images; the browser replaces the
displayed image each time a new one arrives, making it look like video.

---

## 2 — The Files

```
yolo-fun/
├── app.py                    ← The main Python program (the server + AI)
├── requirements.txt          ← Lists the Python packages we need
├── yolo11n_ncnn_model/       ← The AI model files (you create this once)
└── templates/
    └── index.html            ← The web page your browser loads
```

---

## 3 — `requirements.txt` (the shopping list)

```text
flask>=3.0,<4.0
opencv-python-headless>=4.8,<5.0
ultralytics>=8.3,<9.0
```

Before running the app you say `pip install -r requirements.txt` and
Python downloads these three libraries:

| Package | What it does |
|---|---|
| **Flask** | A lightweight web-server framework — lets us say *"when someone visits this URL, give them this response"* |
| **opencv-python-headless** | Computer-vision library — we use it for colour conversion, JPEG encoding, and drawing boxes. The `-headless` suffix means it skips GUI/display code the Pi doesn't need |
| **ultralytics** | The official YOLO library — loads the AI model and runs object detection on images |

### What about Picamera2?

It's **pre-installed** on Raspberry Pi OS. You should **not** pip-install
it. It comes from the system package manager (`apt install python3-picamera2`).

### Version range syntax

`>=3.0,<4.0` means *"any version from 3.0 up to (but not including) 4.0"*.
This ensures you get a compatible version without locking to one exact number.

---

## 4 — `app.py` — Imports

```python
import atexit                                  # Line 23
import cv2                                     # Line 24
from flask import Flask, Response, render_template  # Line 25
from picamera2 import Picamera2                # Line 26
from ultralytics import YOLO                   # Line 27
```

### What is `import`?

Python code is organized into **modules** (files) and **packages**
(folders of files). `import` tells Python: *"Go find this other file
and make its code available to me."*

### Each import explained

| Import | What it gives us | Where it comes from |
|---|---|---|
| `atexit` | A way to register "run this function when the program exits" | Python's standard library (built in, always available) |
| `cv2` | OpenCV's Python interface — image manipulation, drawing, encoding | The `opencv-python-headless` pip package |
| `Flask` | A class that *is* the web server — you create an instance of it | The `flask` pip package |
| `Response` | A class that represents an HTTP response — we use it to build the streaming reply | Same `flask` package |
| `render_template` | A function that reads an HTML file from `templates/`, fills in any variables, and returns the result as a string | Same `flask` package |
| `Picamera2` | A class that controls the physical camera hardware | Pre-installed system package on the Pi |
| `YOLO` | A class that loads a YOLO AI model and can run detection on images | The `ultralytics` pip package |

### `from X import Y` vs `import X`

- `import cv2` — imports the whole module; you access its contents
  as `cv2.something`.
- `from flask import Flask` — reaches *into* the `flask` module and
  grabs just the `Flask` name, so you can use it directly without the
  `flask.` prefix.

Both are valid; the `from` style is just more convenient when you know
exactly which pieces you need.

---

## 5 — `app.py` — Configuration Constants

```python
JPEG_QUALITY = 80         # Line 32
FRAME_WIDTH  = 1280       # Line 33
FRAME_HEIGHT = 720        # Line 34
```

### Why ALL_CAPS?

In Python, writing a variable name in `ALL_CAPS` is a **convention**
(not a rule the language enforces) that says: *"This is a constant —
set it once and don't change it while the program runs."* It makes
settings easy to find at the top of the file.

| Constant | Meaning |
|---|---|
| `JPEG_QUALITY = 80` | When we compress a frame to JPEG, use quality level 80 out of 100. Lower = smaller file = faster streaming but blurrier. Higher = bigger file = sharper but slower |
| `FRAME_WIDTH = 1280` | We ask the camera for 1280 pixels wide (720p HD) |
| `FRAME_HEIGHT = 720` | We ask the camera for 720 pixels tall |

---

## 6 — `app.py` — Camera Initialisation

```python
picam2 = Picamera2()                                          # Line 39

config = picam2.create_video_configuration(                   # Line 43
    main={"format": "BGR888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
)
picam2.configure(config)                                      # Line 46
picam2.start()                                                # Line 47
```

Let's break this down statement by statement:

### `picam2 = Picamera2()`

This creates a **camera object** — a chunk of code that knows how to
talk to the Arducam hardware. The trailing `()` means "call the
constructor" (the function that creates a new instance). Think of it
like pressing the "on" button.

After this line, `picam2` is a variable pointing to that camera object.
You'll use this variable everywhere you want to interact with the camera.

### `create_video_configuration(...)`

This builds a **settings dictionary** that tells the camera *how* to
deliver frames:

```python
main={"format": "BGR888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
```

- **`main={...}`** — configures the main (full-resolution) output
  stream. Picamera2 supports secondary streams too, but we only need one.
- **`"format": "BGR888"`** — tells the camera hardware's ISP (Image
  Signal Processor) to output each pixel as three bytes:
  **B**lue, **G**reen, **R**ed. This is the colour order OpenCV expects.
  "888" means 8 bits per channel = 256 brightness levels per colour.
- **`"size": (1280, 720)`** — the resolution tuple (width, height).

### `picam2.configure(config)`

Applies those settings to the hardware. The camera driver now knows what
format and resolution we want.

### `picam2.start()`

Begins continuous capture. From this moment on, the camera's ISP is
constantly filling a buffer in memory with fresh frames. We haven't
*read* any frames yet — we've just told the camera to start producing
them.

### `print(f"[camera] ...")`

```python
print(f"[camera] Picamera2 started  "
      f"resolution={FRAME_WIDTH}×{FRAME_HEIGHT}  format=BGR888")
```

This prints a startup message to the terminal. The `f"..."` is an
**f-string** — a Python feature where anything inside `{braces}` gets
replaced with its value. So `{FRAME_WIDTH}` becomes `1280`.

---

## 7 — `app.py` — YOLO Model Loading

```python
MODEL_PATH       = "yolo11n_ncnn_model"       # Line 55
CONFIDENCE_THRESH = 0.50                      # Line 56
PERSON_CLASS_ID   = 0                         # Line 57

model = YOLO(MODEL_PATH)                      # Line 59
```

### What each constant means

| Constant | Value | Purpose |
|---|---|---|
| `MODEL_PATH` | `"yolo11n_ncnn_model"` | The folder on disk containing the exported YOLO model files. The `n` in `yolo11n` means "nano" — the smallest/fastest variant, ideal for the Pi's limited CPU |
| `CONFIDENCE_THRESH` | `0.50` | Only draw a box if the AI is at least 50% confident it sees a person. Prevents flicker from uncertain guesses |
| `PERSON_CLASS_ID` | `0` | In the COCO dataset (the training data YOLO learned from), class ID 0 is "person". There are 80 classes total (car, dog, chair, etc.) — we only care about people |

### `model = YOLO(MODEL_PATH)`

This line does a **lot** of heavy lifting:
1. Reads the model weights and architecture from the `yolo11n_ncnn_model/` folder
2. Loads them into memory
3. Prepares the ncnn inference engine (a lightweight neural-network
   runtime optimised for ARM CPUs like the Pi's)

We do this **once, at startup** rather than per-frame, because loading a
model takes seconds and uses significant memory. Reloading it every
frame would crash the Pi.

### Why is this outside any function?

Code at the "top level" of a Python file (not indented inside a
`def` or `class`) runs **once** when the file is first executed. That's
exactly what we want: load the model once, reuse it forever.

---

## 8 — `app.py` — Camera Cleanup (`atexit`)

```python
def _release_camera() -> None:                 # Line 67
    try:
        picam2.stop()
        picam2.close()
        print("[camera] released")
    except Exception:
        pass

atexit.register(_release_camera)               # Line 80
```

### Why we need this

The camera is a **physical device**. Only one program at a time can use
it. If our program crashes without telling the camera "I'm done", the
Pi's kernel still thinks the camera is in use. You'd have to reboot.
This cleanup code prevents that.

### Line-by-line breakdown

#### `def _release_camera() -> None:`

- **`def`** — defines a new function.
- **`_release_camera`** — the function's name. The leading underscore
  `_` is a Python convention meaning *"this is private / internal — not
  meant to be called from outside this file"*.
- **`() -> None`** — this function takes **no arguments** and returns
  **nothing** (`None`). The `-> None` is a **type hint** — it doesn't
  change behaviour, it's just documentation for humans reading the code.

#### `try: ... except Exception: pass`

This is Python's **error handling** syntax:

```
try:
    # code that might fail
except Exception:
    # what to do if it fails
    pass          ← "do nothing, just ignore the error"
```

Why ignore errors here? If the camera was never started, or was already
closed, calling `.stop()` would crash. But we're shutting down anyway,
so there's nothing useful we can do with the error. `pass` means
"carry on silently".

#### `picam2.stop()` and `picam2.close()`

- `.stop()` — tells the ISP to stop producing frames.
- `.close()` — releases the hardware device entirely, freeing the
  kernel lock.

#### `atexit.register(_release_camera)`

**`atexit.register()`** says: *"When the Python interpreter is about to
exit — for any reason — call this function first."*

Notice we pass `_release_camera` **without parentheses**. We're handing
the function itself to `atexit`, not calling it. `atexit` stores it and
calls it later at exit time.

| Syntax | Meaning |
|---|---|
| `_release_camera()` | **Call** the function right now |
| `_release_camera` | **Reference** the function object (don't call it yet) |

---

## 9 — `app.py` — `process_frame()` (the brain)

This is the most complex function. Let's take it section by section.

### Function signature

```python
def process_frame(frame):
```

Takes one argument: `frame`, which is a **NumPy array** — a giant 3D
grid of numbers representing pixel colours. Its shape is
`(720, 1280, 3)`:
- 720 rows of pixels (height)
- 1280 columns of pixels (width)
- 3 colour values per pixel (Blue, Green, Red — 0 to 255 each)

### Step 1 — Colour Fix

```python
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
```

**The problem:** Even though we asked for `BGR888`, the Picamera2
hardware actually delivers pixels in **RGB** order (Red, Green, Blue).
OpenCV **expects** BGR (Blue, Green, Red). If you skip this conversion,
reds and blues swap — you look like a Smurf.

**`cv2.cvtColor()`** is OpenCV's "convert colour" function:
- First argument: the source image
- Second argument: a constant describing the conversion.
  `cv2.COLOR_RGB2BGR` means *"swap the Red and Blue channels"*.

The result overwrites `frame` with the corrected version.

### Step 2 — YOLO Inference

```python
results = model(frame, verbose=False)
```

This single line runs the entire neural network on the image:
1. Resizes the frame to the model's internal input size (usually 640×640)
2. Feeds it through all the neural-network layers
3. Produces a list of **detections**: each detection has a class ID
   (what object), a confidence score (how sure), and bounding-box
   coordinates (where in the image)

`verbose=False` prevents YOLO from printing timing stats for every
single frame (which floods the terminal).

`results` is a **list**. Usually it has one element per image you passed
in (we passed one image, so it's a list of length 1).

### Step 3–4 — Filtering and Drawing

```python
for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
```

#### The outer loop: `for result in results:`

Iterates over each result (we only have one image, so this loop runs
once). A `for` loop in Python says *"for each item in this collection,
do the following"*.

#### The inner loop: `for box in result.boxes:`

`result.boxes` contains all detected objects in the frame. Each `box` is
one detection. If YOLO found 3 people and a dog, there would be 4 boxes.

#### Extracting class and confidence

```python
cls_id = int(box.cls[0])     # e.g., 0 = person, 16 = dog
conf   = float(box.conf[0])  # e.g., 0.87 = 87% confident
```

- `box.cls` is a tensor (a fancy array) containing the class ID. `[0]`
  grabs the first (and only) value. `int()` converts it from a tensor
  to a plain Python integer.
- `box.conf` is a tensor of the confidence score. `float()` converts it
  to a plain Python decimal number.

#### The filter

```python
if cls_id != PERSON_CLASS_ID or conf < CONFIDENCE_THRESH:
    continue
```

**`continue`** means *"skip the rest of this loop iteration and move on
to the next box"*. So if the detected object isn't a person (`!= 0`),
or if the confidence is below 50% (`< 0.50`), we ignore it completely.

The `!=` operator means "not equal to". The `or` keyword means "if
either condition is true".

#### Extracting coordinates

```python
x1, y1, x2, y2 = map(int, box.xyxy[0])
```

This is dense — let's unpack it:

- `box.xyxy[0]` — the bounding box coordinates in
  **(x1, y1, x2, y2)** format:

  ```
  (x1, y1) ────────────────┐
       │                    │
       │     PERSON         │
       │                    │
       └────────────────── (x2, y2)
  ```

  `x1, y1` = top-left corner. `x2, y2` = bottom-right corner. These are
  pixel coordinates (e.g. x1=200, y1=50, x2=400, y2=600).

- `map(int, ...)` — applies `int()` to each of the four values,
  converting them from floating-point tensors to plain integers (you need
  integers for pixel coordinates — there's no pixel 200.7).

- `x1, y1, x2, y2 = ...` — **tuple unpacking**: Python lets you assign
  multiple variables at once from a collection. The four values pop out
  into four separate variables.

#### Drawing the rectangle

```python
cv2.rectangle(frame, (x1, y1), (x2, y2),
              color=(0, 255, 0), thickness=2)
```

- `cv2.rectangle()` draws a rectangle directly onto the `frame` array
  (modifying it in place).
- `(x1, y1)` — top-left corner.
- `(x2, y2)` — bottom-right corner.
- `color=(0, 255, 0)` — colour in BGR format:
  B=0, G=255, R=0 = **bright green**.
- `thickness=2` — the line is 2 pixels wide.

#### Drawing the confidence label

```python
label = f"Person {conf:.0%}"
```

An f-string with a **format spec**: `{conf:.0%}` means "take `conf`
(e.g. 0.87), multiply by 100, round to 0 decimal places, and add a `%`
sign". Result: `"Person 87%"`.

```python
(tw, th), baseline = cv2.getTextSize(
    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
)
```

`getTextSize()` measures how many pixels the text will take up *before*
we draw it. Returns:
- `(tw, th)` — text width and height in pixels
- `baseline` — extra space below the text baseline (we don't use it)

We need these measurements to draw a filled background rectangle behind
the text so it's readable against the video.

```python
cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1),
              color=(0, 255, 0), thickness=-1)
```

Draws a **filled** rectangle (`thickness=-1` means "fill it in, don't
just outline it") in green, positioned just *above* the bounding box.
This is the label's background.

```python
cv2.putText(frame, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 0), 2)
```

Draws the text on top of that green rectangle:
- `(x1, y1 - 5)` — position (slightly above the box's top edge)
- `cv2.FONT_HERSHEY_SIMPLEX` — a simple sans-serif font built into
  OpenCV
- `0.6` — font scale (60% of the font's default size)
- `(0, 0, 0)` — text colour: black (so it stands out on green)
- `2` — thickness of the font strokes

#### Return

```python
return frame
```

Hands back the annotated BGR frame to whoever called `process_frame()`.
The frame now has green boxes and labels drawn on it.

---

## 10 — `app.py` — `generate_frames()` (the heartbeat)

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
value, it:

1. Resumes from where it last `yield`ed
2. Runs until it hits another `yield`
3. Pauses and hands out that value

Since our generator has `while True:` (an infinite loop), it never stops.
Every time Flask needs the next chunk of data to send to the browser, it
asks this generator, which captures one frame, processes it, encodes it,
and yields it.

### Line by line

#### `encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]`

A list telling `cv2.imencode` which compression settings to use. It's a
key-value pair in list form: the key is `IMWRITE_JPEG_QUALITY`, the
value is `80`. Created once outside the loop so we don't rebuild it every
frame.

#### `while True:`

Loop forever. This is intentional — we want to stream until the server
is stopped.

#### `frame = picam2.capture_array()`

Grabs the latest frame from the camera's buffer as a NumPy array. This
is a **blocking call** — if the camera hasn't produced a new frame yet,
it waits until one is ready.

#### `frame = process_frame(frame)`

Sends the frame through our processing pipeline (colour fix → YOLO →
annotations). Returns the annotated BGR frame.

#### `ok, buffer = cv2.imencode(".jpg", frame, encode_params)`

Compresses the raw pixel array into JPEG format:
- `".jpg"` — the target format
- `frame` — the pixel array (720 × 1280 × 3 bytes = ~2.7 MB)
- `encode_params` — JPEG quality 80

Returns two values (**tuple unpacking** again):
- `ok` — a boolean: `True` if encoding succeeded, `False` if it failed
- `buffer` — the compressed JPEG data as a NumPy array of bytes
  (typically ~50–100 KB — much smaller than the raw 2.7 MB)

#### `if not ok: continue`

If encoding failed for some reason, skip this frame and try the next
one. `not` flips `True` to `False` and vice versa.

#### The `yield` block

```python
yield (
    b"--frame\r\n"
    b"Content-Type: image/jpeg\r\n\r\n"
    + buffer.tobytes()
    + b"\r\n"
)
```

This yields one MJPEG "chunk". Let's decode the syntax:

**`b"..."`** — a **bytes literal**. Regular strings in Python are
Unicode text. Bytes literals are raw binary data, which is what HTTP
transmits.

**`\r\n`** — a "carriage return + line feed", the standard line ending
in HTTP. Think of it as "press Enter" in web protocol language.

**`buffer.tobytes()`** — converts the NumPy byte array into a plain
Python `bytes` object that we can concatenate with `+`.

The full chunk, laid out visually:

```
--frame\r\n                          ← "Here comes a new part"
Content-Type: image/jpeg\r\n\r\n     ← "It's a JPEG image" + blank line
[...87,000 bytes of JPEG data...]    ← The actual compressed image
\r\n                                 ← End of this part
```

The browser receives this, displays the JPEG, then waits for the next
`--frame` boundary. When it arrives, the browser **replaces** the image.
This happens ~15–30 times per second → looks like live video.

---

## 11 — `app.py` — Flask Routes (the web server)

### Creating the app

```python
app = Flask(__name__)
```

- `Flask(...)` creates a new web-server application.
- `__name__` is a special Python variable that equals the current
  module's name (in this case, `"__main__"` since we're running the file
  directly). Flask uses it to locate the `templates/` folder.

### What is `@app.route(...)`?

```python
@app.route("/")
def index():
    return render_template("index.html")
```

The `@` symbol is a **decorator** — a special syntax that wraps a
function with extra behaviour. `@app.route("/")` tells Flask: *"When
any browser visits the root URL (`/`), call this function and send its
return value back as the HTTP response."*

- **`"/"`** — the root URL (homepage). If the Pi's IP is `192.168.1.42`,
  then visiting `http://192.168.1.42:5000/` triggers this function.
- **`render_template("index.html")`** — reads the file
  `templates/index.html`, and returns it as a string.
  Flask automatically looks
  inside the `templates/` folder.

### The streaming route

```python
@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
```

When a browser requests `/video_feed` (because the `<img>` tag's `src`
points there), Flask calls `video_feed()` which returns a `Response`
object:

- **`generate_frames()`** — the generator function from above.
  Flask calls `next()` on it repeatedly, sending each yielded chunk to
  the browser.
- **`mimetype="multipart/x-mixed-replace; boundary=frame"`** — an HTTP
  header that tells the browser:
  - `multipart` — "I'll be sending multiple parts."
  - `x-mixed-replace` — "Each new part **replaces** the previous one."
  - `boundary=frame` — "The string `--frame` separates the parts."

  This is what makes the browser **swap** images instead of stacking
  them. It's the core trick that enables MJPEG streaming.

---

## 12 — `app.py` — Entry Point (`if __name__ == "__main__"`)

```python
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        _release_camera()
```

### `if __name__ == "__main__":`

Every Python file has a hidden variable called `__name__`. When you run
a file **directly** (`python app.py`), Python sets `__name__` to the
string `"__main__"`. When a file is **imported** by another file,
`__name__` is set to the file's module name instead.

This `if` block says: *"Only start the web server if someone ran this
file directly. If someone just imported it, don't auto-start."*

### `app.run(...)`

Starts Flask's built-in web server:

| Argument | Meaning |
|---|---|
|`host="0.0.0.0"` | Listen on **all** network interfaces. Without this, Flask only listens on `127.0.0.1` (localhost), which means only the Pi itself could see the page — no other device on the Wi-Fi could reach it |
| `port=5000` | The "door number" in the IP address. Browsers will connect to `http://<pi-ip>:5000` |
| `threaded=True` | Create a new **thread** for each incoming request. This lets multiple browser tabs or devices watch the stream simultaneously |

### `try: ... finally:`

```python
try:
    app.run(...)    # runs until you hit Ctrl-C or it crashes
finally:
    _release_camera()
```

`finally` is the guarantee block. No matter *how* `app.run()` ends —
normal shutdown, Ctrl-C, crash, exception — the code inside `finally:`
**always** runs. This is our safety net to release the camera hardware.

This is the "belt-and-suspenders" approach: `atexit` is the belt,
`finally` is the suspenders. Both do the same thing. If one fails, the
other catches it.

---

## 13 — `templates/index.html` (the web page)

The functionally important line:

```html
<img id="live-stream" src="/video_feed" alt="Live camera stream" />
```

| Attribute | What it does |
|---|---|
| `id="live-stream"` | A unique name for this element (useful for JavaScript later) |
| `src="/video_feed"` | Tells the browser to make a request to `/video_feed` — our MJPEG streaming route. The browser treats the response as the image source. Because the response is `multipart/x-mixed-replace`, the browser keeps swapping in new images |
| `alt="Live camera stream"` | Text shown if the image can't load (accessibility) |

Everything else in the HTML is styling:
- Dark background (`#0b0f19`)
- Gradient title text
- A pulsing red "Live" badge using CSS `@keyframes` animation
- Responsive sizing (`max-width: 90vw`, `max-height: 75vh`)

---

## 14 — How All the Pieces Connect

Here's the full data flow from camera sensor to browser pixel — in
order, every single time a frame is streamed:

```
 ① Camera sensor captures photons → raw Bayer data
                    │
 ② Picamera2's ISP converts Bayer → RGB pixel array (numpy)
                    │
 ③ generate_frames() calls picam2.capture_array()
                    │
 ④ generate_frames() calls process_frame(frame)
        │
        ├─ ⑤ cv2.cvtColor: swap RGB → BGR
        │
        ├─ ⑥ model(frame): YOLO runs 80-layer neural network
        │       → returns list of detections (class, confidence, box)
        │
        ├─ ⑦ Filter: keep only class 0 (person) with conf > 50%
        │
        ├─ ⑧ cv2.rectangle + cv2.putText: draw green boxes & labels
        │
        └─ ⑨ return annotated BGR frame
                    │
 ⑩ cv2.imencode: compress BGR frame → JPEG bytes (~50-100 KB)
                    │
 ⑪ yield: wrap JPEG in MJPEG multipart boundary
                    │
 ⑫ Flask sends bytes over HTTP to the browser
                    │
 ⑬ Browser <img> tag receives the JPEG, displays it, waits for next one
                    │
          [loop back to ① — runs ~15-30 times per second]
```

### Timing on a Raspberry Pi 4B

| Step | Approximate time |
|---|---|
| Camera capture | ~30 ms |
| RGB→BGR conversion | ~2 ms |
| YOLO inference (ncnn, 4-core ARM) | ~80–150 ms |
| Drawing + JPEG encoding | ~5 ms |
| **Total per frame** | **~120–190 ms → about 5–8 FPS** |

5–8 FPS is typical for YOLO Nano on a Pi 4 without a GPU. It's not
silky smooth, but it's "live" enough to be useful for person detection.

---

## 15 — How to Run It

### First time setup (on the Pi)

```bash
# 1. Clone the repo
git clone https://github.com/clayskaggsmagic-ops/yolo-fun.git
cd yolo-fun

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Export the YOLO model to ncnn format (one-time step)
python -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')      # downloads the model weights (~6 MB)
model.export(format='ncnn')     # creates yolo11n_ncnn_model/ folder
"
```

### Running the server

```bash
python app.py
```

You should see:

```
[camera] Picamera2 started  resolution=1280×720  format=BGR888
[yolo]   loaded yolo11n_ncnn_model
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
```

### Viewing the stream

1. Find the Pi's IP: run `hostname -I` (e.g. `192.168.1.42`)
2. On any device on the same Wi-Fi, open a browser
3. Go to `http://192.168.1.42:5000`
4. You should see live video with green boxes around any people

### Stopping

Press **Ctrl+C** in the terminal. The camera will release automatically.

---

## 16 — Glossary

| Term | Meaning |
|---|---|
| **atexit** | Python module that lets you register functions to run when the program exits |
| **BGR** | Blue-Green-Red — the colour channel order OpenCV uses (opposite of RGB) |
| **Bounding box** | A rectangle drawn around a detected object showing its location |
| **bytes literal** | `b"hello"` — raw binary data in Python, as opposed to a text string |
| **Class ID** | A number identifying what type of object YOLO detected (0 = person, 1 = bicycle, etc.) |
| **Confidence score** | A 0.0–1.0 number representing how sure the AI is about a detection |
| **COCO** | Common Objects in Context — the dataset (80 object types) YOLO was trained on |
| **CSI** | Camera Serial Interface — the ribbon cable connecting the camera to the Pi |
| **Decorator** | `@something` syntax that wraps a function with extra behaviour |
| **f-string** | `f"text {variable}"` — Python string that embeds variable values |
| **Flask** | A lightweight Python web-server framework |
| **Frame** | A single still image from the camera. Many frames per second = video |
| **Generator** | A function that uses `yield` to produce values one at a time in a loop |
| **ISP** | Image Signal Processor — hardware on the Pi that converts raw sensor data into pixels |
| **JPEG** | A compressed image format. Much smaller than raw pixels |
| **libcamera** | The Linux camera framework that Picamera2 uses under the hood |
| **MJPEG** | Motion JPEG — streaming video by sending a sequence of JPEGs over HTTP |
| **ncnn** | A lightweight neural-network inference library optimised for ARM CPUs |
| **NumPy array** | A grid of numbers in memory. Images are 3D arrays: height × width × 3 channels |
| **OpenCV (cv2)** | Computer-vision library for image processing, drawing, and encoding |
| **Picamera2** | The official Python library for Raspberry Pi cameras |
| **Route** | A URL pattern (`/`, `/video_feed`) that Flask maps to a Python function |
| **Tensor** | A multi-dimensional array used by AI frameworks (like a numpy array but for models) |
| **Thread** | A lightweight parallel execution path — lets multiple requests run at once |
| **Tuple unpacking** | `a, b = (1, 2)` — assigning multiple variables from a collection in one line |
| **Type hint** | `-> None` or `frame: np.ndarray` — annotations that describe expected types (documentation only) |
| **V4L2** | Video4Linux2 — a Linux kernel interface for cameras. We bypass it due to bugs |
| **YOLO** | "You Only Look Once" — a family of fast AI object-detection models |
