# =========================
# feature_extraction.py
# =========================
import numpy as np
import pandas as pd
import os
import glob

# =========================
# CẤU HÌNH KEYPOINT
# =========================
# Thứ tự khớp trong CSV (theo file detect):
# 0:nose  1:neck  2:left_shoulder  3:right_shoulder
# 4:left_elbow  5:right_elbow  6:left_wrist  7:right_wrist

KEYPOINT_NAMES = [
    "nose", "neck",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist"
]

# =========================
# ĐỌC JOINTS TỪ 1 HÀNG CSV
# =========================
def row_to_joints(row):
    """
    Đọc tọa độ pixel từ 1 hàng DataFrame
    → list 8 điểm [(x0,y0), ..., (x7,y7)]
    """
    joints = []
    for name in KEYPOINT_NAMES:
        x = row[f"{name}_x"]
        y = row[f"{name}_y"]
        joints.append((x, y))
    return joints

# =========================
# COMPONENT 1: NORMALIZED JOINT LOCATION
# Công thức (1): x' = x/W, y' = y/H
# 8 khớp × 2 = 16 features
# =========================
def normalized_joints(joints, img_w, img_h):
    norm = [(x / img_w, y / img_h) for (x, y) in joints]
    return np.array(norm).flatten()   # shape (16,)

# =========================
# COMPONENT 2: JOINT DISTANCES
# Công thức (2): dist(A,B) = sqrt((xB-xA)² + (yB-yA)²)
# 5 khoảng cách d1→d5 (theo paper, bỏ neck↔shoulder):
#   d1 = nose(0)        ↔ neck(1)
#   d2 = neck(1)        ↔ left_shoulder(2)
#   d3 = left_shoulder(2) ↔ left_elbow(4)    [index 4]
#   d4 = neck(1)        ↔ right_shoulder(3)
#   d5 = right_shoulder(3) ↔ right_elbow(5)  [index 5]
# =========================
def euclidean(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def joint_distances(joints):
    j = joints
    d1 = euclidean(j[0], j[1])   # nose  → neck
    d2 = euclidean(j[1], j[2])   # neck  → left_shoulder
    d3 = euclidean(j[2], j[4])   # left_shoulder → left_elbow
    d4 = euclidean(j[1], j[3])   # neck  → right_shoulder
    d5 = euclidean(j[3], j[5])   # right_shoulder → right_elbow
    return np.array([d1, d2, d3, d4, d5])   # shape (5,)

# =========================
# COMPONENT 3: BONE ANGLES
# Góc giữa vector xương và trục tham chiếu (v0, v90, v180)
#
# φ1: vector neck→nose (v10)        với v90  (trục dọc 90°)
# φ2: vector right_shoulder→neck (v_right_sh→neck = v32 theo paper)
#        với v180 (trục ngang 180°)
# φ3: vector right_elbow→right_wrist (v43 theo paper)
#        với v90
# φ4: vector left_shoulder→left_elbow (v65 theo paper)
#        với v0  (trục ngang 0°)
# φ5: vector left_elbow→left_wrist (v76 theo paper)
#        với v90
#
# NOTE: index trong paper dùng số thứ tự của chính paper (0-7),
#       map sang index mảng joints của chúng ta:
#   paper-0 = nose(0), paper-1 = neck(1),
#   paper-2 = left_shoulder(2), paper-3 = left_elbow(4),
#   paper-4 = left_wrist(6), paper-5 = right_shoulder(3),
#   paper-6 = right_elbow(5), paper-7 = right_wrist(7)
# =========================
def angle_with_ref(vec, ref_deg):
    """
    Tính góc (độ) giữa vector 2D và trục tham chiếu.
    ref_deg: 0 = ngang phải, 90 = dọc lên, 180 = ngang trái
    """
    ref_rad = np.deg2rad(ref_deg)
    ref_vec = np.array([np.cos(ref_rad), -np.sin(ref_rad)])  # y lật (ảnh)

    norm_v = np.linalg.norm(vec)
    if norm_v < 1e-6:
        return 0.0

    vec_unit = vec / norm_v
    cos_val = np.clip(np.dot(vec_unit, ref_vec), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))

def bone_angles(joints):
    j = joints

    # φ1: neck(1) → nose(0) , so with v90
    v10  = np.array([j[0][0] - j[1][0], j[0][1] - j[1][1]])
    phi1 = angle_with_ref(v10, 90)

    # φ2: right_elbow(5) → right_shoulder(3) , so with v180
    v32  = np.array([j[3][0] - j[5][0], j[3][1] - j[5][1]])
    phi2 = angle_with_ref(v32, 180)

    # φ3: right_shoulder(3) → right_wrist(7) , so with v90
    v43  = np.array([j[7][0] - j[3][0], j[7][1] - j[3][1]])
    phi3 = angle_with_ref(v43, 90)

    # φ4: left_elbow(4) → left_shoulder(2) , so with v0
    v65  = np.array([j[2][0] - j[4][0], j[2][1] - j[4][1]])
    phi4 = angle_with_ref(v65, 0)

    # φ5: left_elbow(4) → left_wrist(6) , so with v90
    v76  = np.array([j[6][0] - j[4][0], j[6][1] - j[4][1]])
    phi5 = angle_with_ref(v76, 90)

    return np.array([phi1, phi2, phi3, phi4, phi5])   # shape (5,)

# =========================
# TỔNG HỢP FEATURE VECTOR
# 16 + 5 + 5 = 26 features
# =========================
def extract_feature_vector(joints, img_w=1280, img_h=720):
    """
    joints : list 8 điểm pixel [(x,y), ...]
    img_w/h: kích thước ảnh gốc để normalize
    → np.array shape (26,)
    """
    loc    = normalized_joints(joints, img_w, img_h)   # (16,)
    dist   = joint_distances(joints)                    # (5,)
    angles = bone_angles(joints)                        # (5,)
    return np.concatenate([loc, dist, angles])          # (26,)

# =========================
# XỬ LÝ 1 FILE CSV
# =========================
def process_csv(csv_path, img_w=1280, img_h=720):
    df = pd.read_csv(csv_path)
    rows = []

    for _, row in df.iterrows():
        joints = row_to_joints(row)
        fv = extract_feature_vector(joints, img_w, img_h)

        entry = {"frame_id": row["frame_id"]}

        # 16 normalized joint locations
        for i, name in enumerate(KEYPOINT_NAMES):
            entry[f"norm_{name}_x"] = fv[i * 2]
            entry[f"norm_{name}_y"] = fv[i * 2 + 1]

        # 5 distances
        for i in range(5):
            entry[f"d{i+1}"] = fv[16 + i]

        # 5 angles
        for i in range(5):
            entry[f"phi{i+1}"] = fv[21 + i]

        entry["label"] = row["label"]
        rows.append(entry)

    return pd.DataFrame(rows)

# =========================
# MAIN: XỬ LÝ TẤT CẢ CSV
# =========================
if __name__ == "__main__":
    # Thay đổi nếu cần
    IMG_W = 1280
    IMG_H = 720
    INPUT_DIR  = "."          # thư mục chứa các file data_*.csv
    OUTPUT_FILE = "features.csv"

    csv_files = glob.glob(os.path.join(INPUT_DIR, "data_*.csv"))

    if not csv_files:
        print("Không tìm thấy file data_*.csv nào!")
        exit()

    all_dfs = []
    for path in csv_files:
        label = os.path.basename(path).replace("data_", "").replace(".csv", "")
        print(f"  Đang xử lý: {path}  (label={label})")
        df_feat = process_csv(path, IMG_W, IMG_H)
        all_dfs.append(df_feat)

    result = pd.concat(all_dfs, ignore_index=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print(f"\nHoàn tất! Đã lưu {len(result)} dòng vào '{OUTPUT_FILE}'")
    print(f"Shape: {result.shape}")
    print(result.head())