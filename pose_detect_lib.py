import cv2
import numpy as np
from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# INIT YOLO
# =========================
yolo = YOLO("yolov8n.pt")

# =========================
# INIT MEDIAPIPE TASK API
# =========================
base_options = python.BaseOptions(
    model_asset_path="pose_landmarker_lite.task"
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    num_poses=5
)

landmarker = vision.PoseLandmarker.create_from_options(options)

# LOAD MODEL
model = yolo("yolov8n.pt")

# =========================
# CONVERT LANDMARKS (UPDATED)
# =========================
def convert_landmarks(pose_landmarks, w, h):
    """
    MediaPipe (33) → skeleton + thêm neck & midhip
    """

    # ===== LẤY CÁC ĐIỂM CẦN TRONG SKELETON CUA MEDIAPIPE
    def get_point(idx):
        lm = pose_landmarks[idx]
        return (lm.x * w, lm.y * h)

    nose = get_point(0)

    left_shoulder = get_point(11)
    right_shoulder = get_point(12)

    left_elbow = get_point(13)
    right_elbow = get_point(14)

    left_wrist = get_point(15)
    right_wrist = get_point(16)

    left_hip = get_point(23)
    right_hip = get_point(24)

    # ===== TẠO JOINT ẢO =====
    # Neck
    neck = (
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2
    )

    # MidHip
    midhip = (
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2
    )

    # ===== BUILD SKELETON =====
    skeleton = [
        nose,           # 0
        neck,           # 1 (NEW)
        left_shoulder,  # 2
        right_shoulder, # 3
        left_elbow,     # 4
        right_elbow,    # 5
        left_wrist,     # 6
        right_wrist,    # 7
        midhip,         # 8 (NEW)
        left_hip,       # 9
        right_hip       # 10
    ]

    return skeleton


# =========================
# MAIN DETECT FUNCTION
# =========================
def detect(frame):
    h, w, _ = frame.shape

    # ---- YOLO detect ----
    yolo_res = yolo(frame)[0]
    boxes = []

    for r in yolo_res.boxes.data:
        x1, y1, x2, y2, conf, cls = r
        if int(cls) == 0:
            boxes.append([int(x1), int(y1), int(x2), int(y2)])

    # ---- MediaPipe detect ----
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frameRGB
    )

    results = landmarker.detect(mp_image)

    skeletons = []

    if results.pose_landmarks:
        for pose_landmarks in results.pose_landmarks:
            skeleton = convert_landmarks(pose_landmarks, w, h)
            skeletons.append(skeleton)

    return boxes, skeletons


# =========================
# DRAW (DEBUG)
# =========================
def draw(frame, skeletons):
    for sk in skeletons:
        for (x, y) in sk:
            cv2.circle(frame, (int(x), int(y)), 4, (0,255,0), -1)
    return frame

cap = cv2.VideoCapture(0)  # 0 = webcam

if not cap.isOpened():
    print("Không mở được camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes, skeletons = detect(frame)

    frame = draw(frame, skeletons)

    cv2.imshow("YOLO + MediaPipe Pose", frame)

    # nhấn q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()