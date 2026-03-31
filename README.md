# Face Mask Detection using YOLOv8
Real-time face mask detector trained on 853 annotated images across 3 classes.

## 📊 Analyses Performed
- Real-time webcam detection with bounding boxes
- mAP50 and mAP50-95 evaluation
- Training loss curves
- Precision & Recall metrics

## 🤖 ML Model
- YOLOv8 (nano) custom trained
- 3 Classes: with_mask, without_mask, mask_weared_incorrect
- mAP50: ~0.87
- 80/20 Train-Val Split

## 🛠️ Technologies Used
- Python
- Ultralytics YOLOv8
- OpenCV
- PyYAML

## ▶️ How to Run
pip install ultralytics opencv-python pyyaml
python train.py

## 📁 Output
Best weights saved at runs/train/face_mask/weights/best.pt

## 👩‍💻 Author
Rajshree - ML Engineer | Python Developer
