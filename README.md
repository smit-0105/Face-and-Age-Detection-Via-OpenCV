# Gender & Age Detection using Deep Learning

Real-time face analysis using pre-trained Caffe neural networks via OpenCV DNN.
No model training required — download weights and run in minutes.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download models
```bash
python download_models.py
```
For the two Caffe `.caffemodel` files, follow the manual download links printed by the script and place them in `models/`.

### 3. Run

**Webcam (real-time):**
```bash
python detect.py
```

**Single image:**
```bash
python detect.py --image path/to/photo.jpg
```

**Adjust face detection confidence:**
```bash
python detect.py --conf 0.8
```

---

## Controls (Webcam Mode)
| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot |

---

## How It Works

```
Video Frame
    │
    ▼
OpenCV Face Detector (TF .pb model)
    │  detects bounding boxes
    ▼
Crop each face + add padding
    │
    ├──► Gender Net (Caffe)  →  Male / Female + confidence
    │
    └──► Age Net (Caffe)     →  Age range + confidence
         e.g. (25-32), (38-43)
    │
    ▼
Annotated output with bounding boxes + labels
```

### Age Ranges Predicted
`(0-2)` `(4-6)` `(8-12)` `(15-20)` `(25-32)` `(38-43)` `(48-53)` `(60-100)`

---

## Model Architecture
| Model | Type | Input Size | Backbone |
|-------|------|-----------|----------|
| Face Detector | TF pb | 300×300 | SSD + MobileNet |
| Gender Net | Caffe | 227×227 | AlexNet-style |
| Age Net | Caffe | 227×227 | AlexNet-style |

---

## Project Structure
```
gender_age_detection/
├── detect.py            ← Main detection script
├── download_models.py   ← Model downloader
├── requirements.txt
├── README.md
└── models/
    ├── opencv_face_detector.pbtxt
    ├── opencv_face_detector_uint8.pb
    ├── age_deploy.prototxt
    ├── age_net.caffemodel        ← manual download
    ├── gender_deploy.prototxt
    └── gender_net.caffemodel     ← manual download
```

---

## Known Limitations
- Age is predicted as a range, not an exact number
- Accuracy drops with extreme head poses (>45° tilt)
- Poor lighting significantly affects detection confidence
- Model was trained primarily on western demographic datasets
