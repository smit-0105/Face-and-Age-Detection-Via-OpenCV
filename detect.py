"""
Gender and Age Detection - Multi-Person Version
=================================================
Fixes:
  - Each detected face gets its own independent smoother
  - Faces are tracked by position (IoU matching) across frames
  - Each person shows their own stable gender/age label

Usage:
    python detect.py                    # Webcam
    python detect.py --image photo.jpg  # Image
"""

import cv2
import numpy as np
import argparse
import os
import sys
from collections import Counter

# ── Model Paths ────────────────────────────────────────────────────────────────
MODEL_DIR    = "models"
AGE_PROTO    = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL    = os.path.join(MODEL_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_net.caffemodel")
FACE_PROTO   = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")
FACE_MODEL   = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")

MODEL_MEAN   = (78.4263377603, 87.7689143744, 114.895847746)

AGE_LIST    = ["0-2","4-6","8-12","15-20","25-32","38-43","48-53","60+"]
GENDER_LIST = ["Male", "Female"]
COLORS      = {"Male": (255, 144, 30), "Female": (147, 20, 255)}

# ── Tuning Parameters ──────────────────────────────────────────────────────────
SMOOTH_WINDOW    = 30    # frames per face smoother
PREDICT_EVERY    = 5     # run neural net every N frames
MIN_FACE_SIZE    = 80    # ignore tiny faces
GENDER_THRESHOLD = 0.75  # min confidence for gender
AGE_THRESHOLD    = 0.45  # min confidence for age
PADDING          = 30    # pixels of padding around face
IOU_THRESHOLD    = 0.35  # min IoU to match face to existing tracker


# ── Model Loader ───────────────────────────────────────────────────────────────
def load_models():
    print("[INFO] Loading models...")
    for path in [AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL, FACE_PROTO, FACE_MODEL]:
        if not os.path.exists(path):
            print(f"[ERROR] Missing: {path}")
            sys.exit(1)
    face_net     = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    age_net      = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net   = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    print("[INFO] Models loaded.")
    return face_net, age_net, gender_net


# ── Per-Face Smoother ──────────────────────────────────────────────────────────
class FaceSmoother:
    """Independent smoother for a single tracked face."""
    def __init__(self):
        self.genders       = []
        self.ages          = []
        self.stable_gender = "..."
        self.stable_age    = "..."
        self.last_g_conf   = 0.0
        self.last_box      = None
        self.missed_frames = 0   # frames since last detection

    def update(self, gender, age, g_conf, box):
        self.last_box      = box
        self.last_g_conf   = g_conf
        self.missed_frames = 0

        if gender:
            self.genders.append(gender)
        if age:
            self.ages.append(age)

        # Keep only last SMOOTH_WINDOW entries
        self.genders = self.genders[-SMOOTH_WINDOW:]
        self.ages    = self.ages[-SMOOTH_WINDOW:]

        if self.genders:
            self.stable_gender = Counter(self.genders).most_common(1)[0][0]
        if self.ages:
            self.stable_age = Counter(self.ages).most_common(1)[0][0]

    def get(self):
        return self.stable_gender, self.stable_age, self.last_g_conf


# ── IoU Calculator ─────────────────────────────────────────────────────────────
def iou(boxA, boxB):
    """Intersection over Union between two boxes (x1,y1,x2,y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter)


# ── Face Tracker ───────────────────────────────────────────────────────────────
class FaceTracker:
    """
    Maintains a dict of FaceSmoother objects, one per tracked face.
    Matches new detections to existing smoothers using IoU.
    Removes smoothers that haven't been seen for 15 frames.
    """
    def __init__(self):
        self.smoothers  = {}   # id -> FaceSmoother
        self.next_id    = 0
        self.MAX_MISSED = 15

    def update(self, boxes, age_net, gender_net, frame, frame_count):
        h, w = frame.shape[:2]

        # Mark all existing smoothers as potentially missed
        for sid in list(self.smoothers.keys()):
            self.smoothers[sid].missed_frames += 1

        for box in boxes:
            x1, y1, x2, y2 = box

            # Try to match to existing smoother by IoU
            best_id  = None
            best_iou = IOU_THRESHOLD
            for sid, sm in self.smoothers.items():
                if sm.last_box is not None:
                    score = iou(box, sm.last_box)
                    if score > best_iou:
                        best_iou = score
                        best_id  = sid

            # No match → create new smoother
            if best_id is None:
                best_id = self.next_id
                self.smoothers[best_id] = FaceSmoother()
                self.next_id += 1

            sm = self.smoothers[best_id]
            sm.last_box      = box
            sm.missed_frames = 0

            # Run neural net every PREDICT_EVERY frames
            if frame_count % PREDICT_EVERY == 0:
                x1p = max(0, x1 - PADDING)
                y1p = max(0, y1 - PADDING)
                x2p = min(w, x2 + PADDING)
                y2p = min(h, y2 + PADDING)
                face_roi = frame[y1p:y2p, x1p:x2p]

                if face_roi.size > 0:
                    gender, g_conf, age, _ = predict_age_gender(
                        face_roi, age_net, gender_net
                    )
                    sm.update(gender, age, g_conf, box)
                else:
                    sm.last_box = box
            else:
                sm.last_box = box

        # Remove smoothers not seen for too long
        for sid in list(self.smoothers.keys()):
            if self.smoothers[sid].missed_frames > self.MAX_MISSED:
                del self.smoothers[sid]

    def get_results(self):
        """Returns list of (box, gender, age, g_conf) for all active faces."""
        results = []
        for sm in self.smoothers.values():
            if sm.last_box is not None and sm.missed_frames == 0:
                gender, age, g_conf = sm.get()
                results.append((sm.last_box, gender, age, g_conf))
        return results


# ── Face Preprocessor ─────────────────────────────────────────────────────────
def preprocess_face(face_roi):
    face  = cv2.resize(face_roi, (227, 227))
    lab   = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l     = clahe.apply(l)
    face  = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return face


# ── Age & Gender Prediction ────────────────────────────────────────────────────
def predict_age_gender(face_roi, age_net, gender_net):
    face = preprocess_face(face_roi)
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN, swapRB=False)

    gender_net.setInput(blob)
    g_preds = gender_net.forward()[0]
    g_idx   = g_preds.argmax()
    g_conf  = float(g_preds[g_idx])
    gender  = GENDER_LIST[g_idx] if g_conf >= GENDER_THRESHOLD else None

    age_net.setInput(blob)
    a_preds = age_net.forward()[0]
    a_idx   = a_preds.argmax()
    a_conf  = float(a_preds[a_idx])
    age     = AGE_LIST[a_idx] if a_conf >= AGE_THRESHOLD else None

    return gender, round(g_conf * 100, 1), age, round(a_conf * 100, 1)


# ── Face Detection ─────────────────────────────────────────────────────────────
def detect_faces(face_net, frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append((x1, y1, x2, y2))
    return boxes


# ── Draw Label ─────────────────────────────────────────────────────────────────
def draw_label(frame, box, gender, age, g_conf):
    x1, y1, x2, y2 = box
    color = COLORS.get(gender, (160, 160, 160))
    label = f"{gender} {g_conf:.0f}%  |  Age: {age}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - lh - bl - 10), (x1 + lw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - bl - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ── Webcam Mode ────────────────────────────────────────────────────────────────
def run_webcam(face_net, age_net, gender_net):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    tracker          = FaceTracker()
    frame_count      = 0
    screenshot_count = 0
    prev_tick        = cv2.getTickCount()

    print("[INFO] Running. Press Q to quit, S to screenshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        boxes = detect_faces(face_net, frame)

        # Update tracker — each face gets its own smoother
        tracker.update(boxes, age_net, gender_net, frame, frame_count)

        # Draw each face independently
        results = tracker.get_results()
        for (box, gender, age, g_conf) in results:
            draw_label(frame, box, gender, age, g_conf)

        # FPS counter
        tick      = cv2.getTickCount()
        fps       = cv2.getTickFrequency() / (tick - prev_tick)
        prev_tick = tick

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(results)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Gender & Age Detection  |  Q=Quit  S=Screenshot", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q')]:
            break
        elif key in [ord('s'), ord('S')]:
            fname = f"screenshot_{screenshot_count:03d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Saved {fname}")
            screenshot_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ── Image Mode ─────────────────────────────────────────────────────────────────
def run_image(image_path, face_net, age_net, gender_net):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read: {image_path}")
        sys.exit(1)

    boxes = detect_faces(face_net, frame)
    print(f"[INFO] Found {len(boxes)} face(s).")

    h, w = frame.shape[:2]
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        face_roi = frame[max(0,y1-PADDING):min(h,y2+PADDING),
                         max(0,x1-PADDING):min(w,x2+PADDING)]
        if face_roi.size == 0:
            continue
        gender, g_conf, age, a_conf = predict_age_gender(
            face_roi, age_net, gender_net
        )
        gender = gender or "Unknown"
        age    = age    or "Unknown"
        print(f"  Face {i+1}: {gender} ({g_conf}%), Age: {age} ({a_conf}%)")
        draw_label(frame, box, gender, age, g_conf)

    out = "output_" + os.path.basename(image_path)
    cv2.imwrite(out, frame)
    print(f"[INFO] Saved to {out}")
    cv2.imshow("Result — press any key to close", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Entry Point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to image file")
    args = parser.parse_args()

    face_net, age_net, gender_net = load_models()
    if args.image:
        run_image(args.image, face_net, age_net, gender_net)
    else:
        run_webcam(face_net, age_net, gender_net)


if __name__ == "__main__":
    main()
