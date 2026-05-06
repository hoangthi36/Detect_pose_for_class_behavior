import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import os

yolo = YOLO("yolov8n.pt")


base_options = python.BaseOptions(
    model_asset_path="pose_landmarker_lite.task"
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    num_poses=5
)

landmarker = vision.PoseLandmarker.create_from_options(options)

label = "Boring"
no_of_frames = 400


def convert_landmarks(pose_landmarks, w, h):
    def get_point(idx):
        lm = pose_landmarks[idx]
        return (lm.x * w, lm.y * h)

    nose           = get_point(0)
    left_shoulder  = get_point(11)
    right_shoulder = get_point(12)
    left_elbow     = get_point(13)
    right_elbow    = get_point(14)
    left_wrist     = get_point(15)
    right_wrist    = get_point(16)

    neck = (
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2
    )

    skeleton = [
        nose,            # 0
        neck,            # 1
        left_shoulder,   # 2
        right_shoulder,  # 3
        left_elbow,      # 4
        right_elbow,     # 5
        left_wrist,      # 6
        right_wrist,     # 7
    ]
    return skeleton

KEYPOINT_NAMES = [
    "nose", "neck",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist"
]

EDGES = [
    (0, 1), (1, 2), (2, 4), (4, 6),
    (1, 3), (3, 5), (5, 7),
]

def count_points_in_box(box, skeleton):
    x1, y1, x2, y2 = box
    count = 0
    for (x, y) in skeleton:
        if x1 <= x <= x2 and y1 <= y <= y2:
            count += 1
    return count


def detect(frame):
    h, w, _ = frame.shape

    yolo_res = yolo(frame)[0]
    boxes = []
    for r in yolo_res.boxes.data:
        x1, y1, x2, y2, conf, cls = r
        if int(cls) == 0:
            boxes.append([int(x1), int(y1), int(x2), int(y2)])

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frameRGB)
    results = landmarker.detect(mp_image)

    skeletons = []
    matched_boxes = []

    if results.pose_landmarks:
        for pose_landmarks in results.pose_landmarks:
            skeleton = convert_landmarks(pose_landmarks, w, h)

            best_box = None
            max_points = 0
            for box in boxes:
                c = count_points_in_box(box, skeleton)
                if c > max_points:
                    max_points = c
                    best_box = box

            if best_box is not None and max_points > 0:
                skeletons.append(skeleton)
                matched_boxes.append(best_box)

    return boxes, skeletons, matched_boxes

def draw(frame, skeletons, boxes=None):
    for sk in skeletons:
        for (x, y) in sk:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

        for (i, j) in EDGES:
            if sk[i] is None or sk[j] is None:
                continue
            x1, y1 = sk[i]
            x2, y2 = sk[j]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

    if boxes is not None:
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return frame


def build_row(frame_idx, skeleton, label):
    row = {"frame_id": frame_idx}

    for i, name in enumerate(KEYPOINT_NAMES):
        x, y = skeleton[i]
        row[f"{name}_x"] = round(x, 3)
        row[f"{name}_y"] = round(y, 3)

    row["label"] = label
    return row

#main
frame_data = []
output_file = f"data_{label}.csv"
frame_idx = 0
collected = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được camera")
    exit()

print(f"Bắt đầu thu thập — nhãn: '{label}' | mục tiêu: {no_of_frames} frame")
print("Nhấn [q] để dừng sớm")

while collected < no_of_frames:
    ret, frame = cap.read()
    if not ret:
        break

    boxes, skeletons, matched_boxes = detect(frame)

    if len(skeletons) > 0:
        for sk in skeletons:
            row = build_row(frame_idx, sk, label)
            frame_data.append(row)
        collected += 1

        cv2.putText(frame,
                    f"{label}  {collected}/{no_of_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 200, 255), 2)

    frame = draw(frame, skeletons, boxes)
    cv2.imshow("Thu thap du lieu", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Dừng sớm theo yêu cầu.")
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

#ghi csv
if frame_data:
    df = pd.DataFrame(frame_data)

    write_header = not os.path.exists(output_file)
    df.to_csv(output_file, mode='a', header=write_header, index=False)

    print(f"\nĐã lưu {len(frame_data)} dòng vào '{output_file}'")
    print(df.head())
else:
    print("Không có dữ liệu nào được thu thập.")