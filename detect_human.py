from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cam =cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Dự đoán pose
    results = model(frame)

    # Vẽ kết quả lên frame
    annotated_frame = results[0].plot()

    # Hiển thị
    cv2.imshow("YOLO Pose - Webcam", annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
