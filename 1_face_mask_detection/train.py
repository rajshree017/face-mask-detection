"""
Face Mask Detection using YOLOv8
---------------------------------
Dataset: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
"""

from ultralytics import YOLO
import yaml, os

# ── Config ──────────────────────────────────────────────────
DATA_YAML   = "data.yaml"
MODEL       = "yolov8n.pt"   # nano – fast training; swap to yolov8s.pt for better accuracy
EPOCHS      = 50
IMG_SIZE    = 640
BATCH       = 16
PROJECT_DIR = "runs/train"

# ── data.yaml (auto-created if missing) ─────────────────────
if not os.path.exists(DATA_YAML):
    cfg = {
        "path": "dataset",          # root folder with images/ and labels/
        "train": "images/train",
        "val":   "images/val",
        "nc":    3,
        "names": ["with_mask", "without_mask", "mask_weared_incorrect"],
    }
    with open(DATA_YAML, "w") as f:
        yaml.dump(cfg, f)
    print(f"[INFO] Created {DATA_YAML} – update 'path' to your dataset location.")

# ── Train ────────────────────────────────────────────────────
def train():
    model = YOLO(MODEL)
    results = model.train(
        data    = DATA_YAML,
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH,
        project = PROJECT_DIR,
        name    = "face_mask",
        exist_ok= True,
    )
    print("\n✅ Training complete!")
    print(f"   Best weights: {PROJECT_DIR}/face_mask/weights/best.pt")
    return results

# ── Evaluate ─────────────────────────────────────────────────
def evaluate(weights="runs/train/face_mask/weights/best.pt"):
    model  = YOLO(weights)
    metrics = model.val(data=DATA_YAML)
    print(f"\nmAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")

# ── Webcam inference ─────────────────────────────────────────
def webcam_demo(weights="runs/train/face_mask/weights/best.pt"):
    model = YOLO(weights)
    model.predict(source=0, show=True, conf=0.4)   # source=0 → webcam

if __name__ == "__main__":
    train()
    evaluate()
