# YOLO-Fun — Setup Guide

Exact commands to get the MJPEG + YOLO streaming server running
on a **Raspberry Pi 4B** from a fresh clone.

---

## Prerequisites

- Raspberry Pi 4B running **64-bit Raspberry Pi OS (Bookworm)**
- Arducam IMX708 (Pi Camera Module 3) connected via CSI ribbon
- Pi connected to your local Wi-Fi network
- A second computer (Mac/PC) for exporting the YOLO model

---

## Step 1 — Reboot the Pi

```bash
sudo reboot
```

Wait ~30 seconds, then SSH back in.

---

## Step 2 — Install system dependencies

> **⚠️ Use APT, not pip.** Pip wheels for aarch64 are compiled with
> ARMv8.2 instructions that cause SIGILL on the Pi 4's ARMv8.0 CPU.
> APT packages are compiled for your exact hardware.

```bash
sudo apt update
sudo apt install -y python3-flask python3-opencv python3-numpy python3-picamera2 git
```

---

## Step 3 — Verify imports

```bash
python3 -c "import cv2; import numpy; import flask; from picamera2 import Picamera2; print('ALL IMPORTS OK')"
```

Should print `ALL IMPORTS OK` with no errors.

---

## Step 4 — Clone the repo

```bash
cd ~
git clone https://github.com/clayskaggsmagic-ops/yolo-fun.git
cd yolo-fun
```

If re-deploying (repo already exists):

```bash
cd ~/yolo-fun
git pull origin main
```

---

## Step 5 — Get the YOLO model

The Pi can't run `ultralytics` (requires PyTorch), so export the model
on a **Mac or PC**, then copy it over.

### On your Mac/PC:

```bash
# Create a temp workspace
mkdir ~/yolo-export && cd ~/yolo-export
python3 -m venv venv && source venv/bin/activate
pip install ultralytics

# Export YOLOv8n to ONNX
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"

# Copy to the Pi (replace <pi-ip> with your Pi's IP or hostname)
scp yolov8n.onnx claybeboy@raspberrypi.local:~/yolo-fun/
```

### Verify on the Pi:

```bash
cd ~/yolo-fun
python3 -c "import cv2; net = cv2.dnn.readNetFromONNX('yolov8n.onnx'); print('MODEL LOADED OK')"
```

---

## Step 6 — Remove leftover pip packages (if any)

If you previously installed packages via pip, remove them so they don't
shadow the apt versions:

```bash
pip uninstall -y opencv-python-headless opencv-python numpy flask ultralytics torch torchvision 2>/dev/null
pip3 uninstall -y opencv-python-headless opencv-python numpy flask ultralytics torch torchvision 2>/dev/null
```

---

## Step 7 — Clean `app.py` of any stale imports

Make sure the `from ultralytics import YOLO` line is gone:

```bash
cd ~/yolo-fun
sed -i '/from ultralytics import YOLO/d' app.py
```

---

## Step 8 — Run the server

```bash
cd ~/yolo-fun
python3 app.py
```

Expected output:

```
[camera] Picamera2 started  resolution=640×480  format=BGR888  (threaded)
[yolo]   loaded yolov8n.onnx  (cv2.dnn / ONNX, input=320)
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
```

---

## Step 9 — View the stream

1. Find the Pi's IP: `hostname -I`
2. On any device on the same Wi-Fi, open a browser
3. Go to **`http://<pi-ip>:5000`**
4. Live video with green person-detection boxes should appear

---

## Step 10 — Stop the server

Press **Ctrl + C** in the SSH terminal. The camera releases
automatically.

---

## Performance Tuning

Settings are at the top of `app.py`. Adjust and restart:

| Setting | Default | Faster ↑ | Sharper ↑ |
|---|---|---|---|
| `FRAME_WIDTH × FRAME_HEIGHT` | 640×480 | 320×240 | 1280×720 |
| `MODEL_INPUT_SIZE` | 320 | 160 | 640 |
| `JPEG_QUALITY` | 60 | 40 | 80 |
| `YOLO_EVERY_N` | 3 | 5 | 1 |
| `CONFIDENCE_THRESH` | 0.45 | 0.60 | 0.30 |

---

## Quick Reference

| What | Command |
|---|---|
| SSH in | `ssh claybeboy@raspberrypi.local` |
| Start server | `cd ~/yolo-fun && python3 app.py` |
| Stop server | `Ctrl + C` |
| View stream | `http://raspberrypi.local:5000` |
| Check Pi IP | `hostname -I` |
| Pull updates | `cd ~/yolo-fun && git pull origin main` |

---

## Troubleshooting

| Error | Fix |
|---|---|
| `SIGILL` / `Illegal instruction` | You have pip-installed native packages. Run Step 6 to uninstall them |
| `ModuleNotFoundError: ultralytics` | Run Step 7 (`sed` command) to remove the stale import |
| `No module named 'cv2'` | Run Step 2 (`apt install python3-opencv`) |
| `Cannot open camera` | Check CSI ribbon is seated; run `sudo raspi-config` → Interfaces → Camera → Enable |
| `ONNX parse error` | You're using YOLO11 instead of YOLOv8. Re-export with YOLOv8n (Step 5) |
| Camera locked after crash | `sudo reboot` |
