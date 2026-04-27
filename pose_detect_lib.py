import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# init models
yolo = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
#vi solution khong con duoc su dung
#Cau hinh base option - tro toi file model .task

# mapping MediaPipe -> OpenPose (18 keypoints)
# (chọn subset gần đúng)
SELECTED_IDS = [
    0,   # nose
    11, 12, 13, 14, 15, 16,  # shoulders, elbows, wrists
    23, 24  # hips (có thể bỏ sau)
]

def detect(frame):
    h, w, _ = frame.shape

    # YOLO detect
    yolo_res = yolo(frame)[0]
    boxes = []

    for r in yolo_res.boxes.data:
        x1, y1, x2, y2, conf, cls = r
        if int(cls) == 0:
            boxes.append([int(x1), int(y1), int(x2), int(y2)])

    # MediaPipe pose
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    skeletons = []
    if res.pose_landmarks:
        joints = []
        for lm in res.pose_landmarks.landmark:
            joints.append((lm.x * w, lm.y * h))

        # chọn subset giống OpenPose
        selected = [joints[i] for i in SELECTED_IDS]
        skeletons.append(selected)

    return boxes, skeletons

def draw_results(frame, boxes, skeletons):
    # vẽ bounding box
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # vẽ keypoints
    for sk in skeletons:
        for (x, y) in sk:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

    return frame


# ===============================
# DEMO WEBCAM
# ===============================
cap = cv2.VideoCapture(0)  # 0 = webcam

if not cap.isOpened():
    print("Không mở được camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes, skeletons = detect(frame)

    frame = draw_results(frame, boxes, skeletons)

    cv2.imshow("YOLO + MediaPipe Pose", frame)

    # nhấn q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()