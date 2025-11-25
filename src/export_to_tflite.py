"""
Export YOLOv8n model to TFLite format.
Requirements:
- ultralytics (pip install ultralytics)

This script assumes `yolov8n.pt` is available or will be downloaded by the ultralytics library.
The exported TFLite file will be saved alongside the model (file name ends with `.tflite`).
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = "yolov8n.pt"


def main():
    model = YOLO(WEIGHTS)
    print("Exporting model to TFLite. This may take a while...")
    model.export(format="tflite")
    print("Export complete.")


if __name__ == '__main__':
    main()
