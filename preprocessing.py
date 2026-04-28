import numpy as np
import pose_detect_lib

# =========================
# UTIL
# =========================
def count_points(box, skeleton):
    x1, y1, x2, y2 = box
    count = 0
    for (x, y) in skeleton:
        if x1 <= x <= x2 and y1 <= y <= y2:
            count += 1
    return count

def belong_to_area(box, joint):
    x1, y1, x2, y2 = box
    x, y = joint
    return x1 <= x <= x2 and y1 <= y <= y2

# =========================
# ALGORITHM 1
# =========================
def match_skeleton_box(skeletons, boxes):
    boxes_skeleton = []

    for body in skeletons:
        xs = [p[0] for p in body]
        ys = [p[1] for p in body]
        boxes_skeleton.append([min(xs), min(ys), max(xs), max(ys)])

    boxes_all = boxes_skeleton.copy()
    status = [True] * len(skeletons)

    for box in boxes:
        index_max = -1
        num = 0

        for i, body in enumerate(skeletons):
            if not status[i]:
                continue
            c = count_points(box, body)
            if c > num:
                num = c
                index_max = i

        if index_max == -1:
            boxes_all.append(box)
        else:
            status[index_max] = False
            boxes_skeleton[index_max] = box
            boxes_all[index_max] = box

    return boxes_skeleton, boxes_all

# =========================
# ALGORITHM 2
# =========================
def clean_skeletons(skeletons, boxes_skeleton):
    new_skeletons = []

    for i, body in enumerate(skeletons):
        new_body = []
        for joint in body:
            if belong_to_area(boxes_skeleton[i], joint):
                new_body.append(joint)
        new_skeletons.append(new_body)

    return new_skeletons

# =========================
# FEATURE EXTRACTION
# =========================
def normalize(joints, w, h):
    return [(x / w, y / h) for (x, y) in joints]

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def extract_features(joints):
    if len(joints) < 7:
        return None

    # chọn 7–8 khớp
    joints = joints[:8]

    # normalize
    joints = normalize(joints, 1, 1)

    loc = np.array(joints).flatten()

    d = [
        dist(joints[0], joints[1]),
        dist(joints[1], joints[2]),
        dist(joints[2], joints[3]),
        dist(joints[1], joints[4]),
        dist(joints[4], joints[5])
    ]

    angles = [0]*5  # demo đơn giản

    return np.concatenate([loc, d, angles])