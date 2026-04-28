import cv2
import torch
from pose_detect_lib import detect
from preprocessing import match_skeleton_box, clean_skeletons, extract_features

from train import Net

model = Net()
model.load_state_dict(torch.load("model.pth"))
model.eval()

labels = ["Asking", "Looking", "Bowing", "Boring"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes, skeletons = detect(frame)

    if skeletons:
        boxes_skel, _ = match_skeleton_box(skeletons, boxes)
        skeletons = clean_skeletons(skeletons, boxes_skel)

        for sk in skeletons:
            feat = extract_features(sk)

            if feat is None or len(feat) != 26:
                continue

            x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                pred = torch.argmax(model(x), dim=1).item()

            label = labels[pred]

            cv2.putText(frame, label, (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(frame, (x1,y1),(x2,y2),(255,0,0),2)

    cv2.imshow("Result", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()