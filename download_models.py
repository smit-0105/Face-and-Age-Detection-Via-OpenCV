"""
download_models.py
==================
Downloads all required pre-trained model files into the models/ folder.
Run this ONCE before running detect.py.

Usage:
    python download_models.py
"""

import os
import urllib.request
import sys

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FILES = {
    # OpenCV face detector (TensorFlow/PB format)
    "opencv_face_detector.pbtxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/opencv_face_detector.pbtxt"
    ),
    "opencv_face_detector_uint8.pb": (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
        "dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb"
    ),
    # Age model
    "age_deploy.prototxt": (
        "https://raw.githubusercontent.com/spmallick/learnopencv/"
        "master/AgeGender/age_deploy.prototxt"
    ),
    "age_net.caffemodel": (
        "https://drive.google.com/uc?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW"
        # Mirror: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age_net.caffemodel
    ),
    # Gender model
    "gender_deploy.prototxt": (
        "https://raw.githubusercontent.com/spmallick/learnopencv/"
        "master/AgeGender/gender_deploy.prototxt"
    ),
    "gender_net.caffemodel": (
        "https://drive.google.com/uc?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ"
        # Mirror: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender_net.caffemodel
    ),
}

# Files that must be manually downloaded (Google Drive requires browser auth)
MANUAL = ["age_net.caffemodel", "gender_net.caffemodel"]

MANUAL_INSTRUCTIONS = """
┌──────────────────────────────────────────────────────────────────────┐
│  MANUAL DOWNLOAD REQUIRED for Caffe model weights                   │
│                                                                      │
│  age_net.caffemodel:                                                 │
│  https://drive.google.com/uc?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW  │
│                                                                      │
│  gender_net.caffemodel:                                              │
│  https://drive.google.com/uc?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ  │
│                                                                      │
│  Save both files to: ./models/                                       │
└──────────────────────────────────────────────────────────────────────┘
"""


def download_file(filename, url):
    dest = os.path.join(MODEL_DIR, filename)
    if os.path.exists(dest):
        print(f"  [SKIP] {filename} already exists.")
        return True

    print(f"  [DOWNLOADING] {filename} ...")
    try:
        def progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(100, block_num * block_size * 100 // total_size)
                sys.stdout.write(f"\r    Progress: {pct}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, dest, progress)
        print(f"\r  [OK] {filename}")
        return True
    except Exception as e:
        print(f"\r  [FAIL] {filename}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def main():
    print("=" * 60)
    print("  Gender & Age Detection — Model Downloader")
    print("=" * 60)

    success = True
    for filename, url in FILES.items():
        if filename in MANUAL:
            continue  # skip auto-download for these
        ok = download_file(filename, url)
        if not ok:
            success = False

    # Check if manual files already exist
    for filename in MANUAL:
        dest = os.path.join(MODEL_DIR, filename)
        if os.path.exists(dest):
            print(f"  [OK] {filename} (already present)")
        else:
            success = False

    if not success:
        print(MANUAL_INSTRUCTIONS)
    else:
        print("\n[DONE] All models ready. Run: python detect.py")


if __name__ == "__main__":
    main()
