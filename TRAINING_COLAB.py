# ============================================================
# TrafficVision AI — YOLOv8 Training Notebook
# Run this in Google Colab (free GPU)
# ============================================================

# CELL 1: Install dependencies
# pip install ultralytics roboflow

# CELL 2: Download dataset from Roboflow
"""
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")

# Download helmet violation dataset
project = rf.workspace("roboflow-universe-projects").project("helmet-detection-iebs0")
dataset = project.version(1).download("yolov8")
"""

# CELL 3: Train YOLOv8
"""
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Start from pretrained

results = model.train(
    data='path/to/dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='traffic_violations',
    patience=10,
    save=True,
    device=0,  # GPU
)

print("Training complete!")
print(f"Best model saved to: {results.save_dir}")
"""

# CELL 4: Validate
"""
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall: {metrics.box.mr:.3f}")
"""

# CELL 5: Download trained weights
"""
# After training, download the best.pt file:
from google.colab import files
files.download('runs/detect/traffic_violations/weights/best.pt')
# Then rename it to traffic_violations.pt and put in models/ folder
"""

# CELL 6: Quick test inference
"""
model = YOLO('runs/detect/traffic_violations/weights/best.pt')
results = model('test_image.jpg', conf=0.4)
results[0].show()
"""
