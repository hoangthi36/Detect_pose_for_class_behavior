# =========================
# inference.py
# =========================
import cv2
import numpy as np
import joblib
from tensorflow import keras

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# =========================
# LOAD MODEL & TOOLS
# =========================
print("Đang load model...")
model  = keras.models.load_model("model.h5")
scaler = joblib.load("scaler.pkl")
le     = joblib.load("label_encoder.pkl")
print(f"Các nhãn: {list(le.classes_)}")

# Màu cho từng nhãn
LABEL_COLORS = {
    "Looking" : (0,   255, 0),
    "Asking"  : (0,   200, 255),
    "Bowing"  : (255, 100, 0),
    "Boring"  : (0,   0,   255),
}
DEFAULT_COLOR = (200, 200, 200)

# =========================
# INIT YOLO
# =========================
yolo = YOLO("yolov8n.pt")

# =========================
# INIT MEDIAPIPE
# =========================
base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    num_poses=5
)
landmarker = vision.PoseLandmarker.create_from_options(options)

# =========================
# CẤU HÌNH SKELETON
# =========================
KEYPOINT_NAMES = [
    "nose", "neck",
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist"
]

EDGES = [
    (0, 1), (1, 2), (2, 4), (4, 6),
    (1, 3), (3, 5), (5, 7),
]

# =========================
# HELPER FUNCTIONS
# =========================
def convert_landmarks(pose_landmarks, w, h):
    def get_point(idx):
        lm = pose_landmarks[idx]
        return (lm.x * w, lm.y * h)

    left_shoulder  = get_point(11)
    right_shoulder = get_point(12)
    neck = (
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2
    )
    return [
        get_point(0),    # 0 nose
        neck,            # 1 neck
        left_shoulder,   # 2
        right_shoulder,  # 3
        get_point(13),   # 4 left_elbow
        get_point(14),   # 5 right_elbow
        get_point(15),   # 6 left_wrist
        get_point(16),   # 7 right_wrist
    ]

def count_points_in_box(box, skeleton):
    x1, y1, x2, y2 = box
    return sum(1 for (x, y) in skeleton if x1 <= x <= x2 and y1 <= y <= y2)

def detect(frame):
    h, w, _ = frame.shape

    # YOLO
    boxes = []
    for r in yolo(frame, verbose=False)[0].boxes.data:
        x1, y1, x2, y2, conf, cls = r
        if int(cls) == 0:
            boxes.append([int(x1), int(y1), int(x2), int(y2)])

    # MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    results = landmarker.detect(mp_image)

    skeletons, matched_boxes = [], []
    if results.pose_landmarks:
        for pose_landmarks in results.pose_landmarks:
            skeleton = convert_landmarks(pose_landmarks, w, h)
            best_box, max_pts = None, 0
            for box in boxes:
                c = count_points_in_box(box, skeleton)
                if c > max_pts:
                    max_pts, best_box = c, box
            if best_box is not None and max_pts > 0:
                skeletons.append(skeleton)
                matched_boxes.append(best_box)

    return boxes, skeletons, matched_boxes

# =========================
# FEATURE EXTRACTION
# =========================
def euclidean(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def angle_with_ref(vec, ref_deg):
    ref_rad = np.deg2rad(ref_deg)
    ref_vec = np.array([np.cos(ref_rad), -np.sin(ref_rad)])
    norm_v  = np.linalg.norm(vec)
    if norm_v < 1e-6:
        return 0.0
    cos_val = np.clip(np.dot(vec / norm_v, ref_vec), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))

def extract_features(skeleton, img_w, img_h):
    j = skeleton

    # 16 normalized joint locations
    loc = np.array([(x / img_w, y / img_h) for (x, y) in j]).flatten()

    # 5 distances
    dist = np.array([
        euclidean(j[0], j[1]),
        euclidean(j[1], j[2]),
        euclidean(j[2], j[4]),
        euclidean(j[1], j[3]),
        euclidean(j[3], j[5]),
    ])

    # 5 bone angles
    v10 = np.array([j[0][0]-j[1][0], j[0][1]-j[1][1]])
    v32 = np.array([j[3][0]-j[5][0], j[3][1]-j[5][1]])
    v43 = np.array([j[7][0]-j[3][0], j[7][1]-j[3][1]])
    v65 = np.array([j[2][0]-j[4][0], j[2][1]-j[4][1]])
    v76 = np.array([j[6][0]-j[4][0], j[6][1]-j[4][1]])

    angles = np.array([
        angle_with_ref(v10,  90),
        angle_with_ref(v32, 180),
        angle_with_ref(v43,  90),
        angle_with_ref(v65,   0),
        angle_with_ref(v76,  90),
    ])

    return np.concatenate([loc, dist, angles])   # (26,)

# =========================
# CLASSIFY 1 NGƯỜI
# =========================
def classify(skeleton, img_w, img_h):
    fv        = extract_features(skeleton, img_w, img_h).reshape(1, -1)
    fv_scaled = scaler.transform(fv)
    probs     = model.predict(fv_scaled, verbose=0)[0]
    idx       = int(np.argmax(probs))
    label     = le.inverse_transform([idx])[0]
    conf      = float(probs[idx])
    return label, conf

# =========================
# DRAW
# =========================
def draw(frame, skeletons, matched_boxes, labels):
    for sk, box, (label, conf) in zip(skeletons, matched_boxes, labels):
        color = LABEL_COLORS.get(label, DEFAULT_COLOR)

        # Vẽ keypoints
        for (x, y) in sk:
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)

        # Vẽ xương
        for (i, j) in EDGES:
            x1, y1 = sk[i]
            x2, y2 = sk[j]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Vẽ bounding box
        bx1, by1, bx2, by2 = box
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

        # Nhãn + confidence
        text = f"{label}  {conf*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (bx1, by1 - th - 10), (bx1 + tw + 6, by1), color, -1)
        cv2.putText(frame, text,
                    (bx1 + 3, by1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return frame

# =========================
# MAIN LOOP
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được camera")
    exit()

print("Inference đang chạy... Nhấn [q] để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    boxes, skeletons, matched_boxes = detect(frame)

    # Phân loại từng người
    labels = [classify(sk, w, h) for sk in skeletons]

    # Vẽ kết quả
    frame = draw(frame, skeletons, matched_boxes, labels)

    # FPS góc trái
    cv2.putText(frame, "Press Q to quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180, 180, 180), 1)

    cv2.imshow("Inference - Behavior Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()