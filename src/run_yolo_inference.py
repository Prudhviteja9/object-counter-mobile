from ultralytics import YOLO
import cv2
import os

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Load pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # List input images
    image_files = [f for f in os.listdir(INPUT_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)
        print(f"Processing {img_path}")

        # Run inference
        results = model(img_path)[0]

        # Count all detections
        num_detections = len(results.boxes)
        print(f"Total objects detected: {num_detections}")

        # Per-class counts
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        counts = {}
        for cid in class_ids:
            counts[cid] = counts.get(cid, 0) + 1

        print("Per-class counts:", counts)

        # Save output image
        plotted = results.plot()  # numpy image
        out_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(out_path, plotted)
        print(f"Saved result to {out_path}\n")

if __name__ == "__main__":
    main()
