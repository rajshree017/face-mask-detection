# 😷 Face Mask Detection — YOLOv8

Real-time face mask detector using YOLOv8 trained on 853 annotated images across 3 classes.

## 📁 Folder Structure
```
1_face_mask_detection/
├── train.py          # Training + evaluation + webcam demo
├── requirements.txt
├── data.yaml         # Auto-generated on first run
└── README.md
```

## 🚀 Setup & Run
```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
# Extract to: dataset/images/train  &  dataset/images/val
# Labels (YOLO format) go in: dataset/labels/train  &  dataset/labels/val

# Train
python train.py

# Webcam demo (after training)
python -c "from train import webcam_demo; webcam_demo()"
```

## 📊 Expected Results
| Metric | Value |
|--------|-------|
| mAP50 | ~0.87 |
| mAP50-95 | ~0.60 |

## 🧠 What You'll Learn
- Custom object detection with YOLOv8
- YOLO label format & data.yaml config
- mAP evaluation metrics
- Real-time inference with webcam
