# =========================
# train.py
# =========================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# =========================
# CẤU HÌNH
# =========================
FEATURE_FILE  = "features.csv"
MODEL_FILE    = "model.h5"
EPOCHS        = 10
BATCH_SIZE    = 32
RANDOM_STATE  = 42

# =========================
# 1. ĐỌC DỮ LIỆU
# =========================
print("=" * 50)
print("1. Đọc dữ liệu...")
df = pd.read_csv(FEATURE_FILE)
print(f"   Tổng số mẫu : {len(df)}")
print(f"   Phân bố nhãn:\n{df['label'].value_counts()}")

# Tách feature và nhãn
feature_cols = [c for c in df.columns if c not in ("frame_id", "label")]
X = df[feature_cols].values.astype(np.float32)   # (N, 26)
y_raw = df["label"].values

# Encode nhãn → số nguyên
le = LabelEncoder()
y = le.fit_transform(y_raw)                       # (N,)
num_classes = len(le.classes_)
print(f"   Số lớp      : {num_classes} → {list(le.classes_)}")

# =========================
# 2. CHIA DỮ LIỆU
# 70% train+val  |  30% test
# train+val → 80/20 → train 56% / val 14% tổng
# =========================
print("\n2. Chia dữ liệu 70/30 ...")
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv,
    test_size=0.20,         # 20% của 70% ≈ 14% tổng
    random_state=RANDOM_STATE,
    stratify=y_tv
)

print(f"   Train : {len(X_train)} mẫu")
print(f"   Val   : {len(X_val)}   mẫu")
print(f"   Test  : {len(X_test)}  mẫu")

# =========================
# 3. CHUẨN HÓA FEATURE
# =========================
print("\n3. Chuẩn hóa dữ liệu (StandardScaler)...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Lưu scaler để dùng lúc inference
import joblib
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le,     "label_encoder.pkl")
print("   Đã lưu scaler.pkl và label_encoder.pkl")

# =========================
# 4. XÂY DỰNG MÔ HÌNH DNN
# Theo paper: Input(26) → FC1(128) → FC2(64) → FC3(16) → FC4(4/softmax)
# Mỗi FC1-FC3: Linear → BatchNorm → ReLU
# =========================
print("\n4. Xây dựng mô hình DNN...")

def build_model(input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,), name="input")

    # FC1 - 128
    x = layers.Dense(128, name="FC1")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # FC2 - 64
    x = layers.Dense(64, name="FC2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # FC3 - 16
    x = layers.Dense(16, name="FC3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # FC4 - output (softmax)
    outputs = layers.Dense(num_classes, activation="softmax", name="FC4")(x)

    model = keras.Model(inputs, outputs)
    return model

model = build_model(input_dim=X_train.shape[1], num_classes=num_classes)
model.summary()

# =========================
# 5. COMPILE
# =========================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# 6. CALLBACKS
# - ModelCheckpoint: lưu model tốt nhất theo val_accuracy
# - ReduceLROnPlateau: giảm lr khi val_loss không cải thiện
# - EarlyStopping: dừng sớm nếu không cải thiện sau 5 epoch
# =========================
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_FILE,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
]

# =========================
# 7. TRAIN
# =========================
print("\n5. Bắt đầu training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# =========================
# 8. ĐÁNH GIÁ TRÊN TEST SET
# =========================
print("\n6. Đánh giá trên Test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"   Test Loss     : {test_loss:.4f}")
print(f"   Test Accuracy : {test_acc * 100:.2f}%")

# Classification report chi tiết
from sklearn.metrics import classification_report, confusion_matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n   Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print(cm_df)

# =========================
# 9. VẼ ĐỒ THỊ TRAINING
# =========================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"],     label="Train Acc")
axes[0].plot(history.history["val_accuracy"], label="Val Acc")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history["loss"],     label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
plt.show()
print("\n   Đã lưu biểu đồ → training_history.png")

# =========================
# TỔNG KẾT
# =========================
print("\n" + "=" * 50)
print(f"  Model tốt nhất → {MODEL_FILE}")
print(f"  Test Accuracy  → {test_acc * 100:.2f}%")
print("=" * 50)