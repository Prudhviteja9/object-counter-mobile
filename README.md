# Object Counter Mobile

A machine learning-based object detection and counting application using YOLOv8. This project provides tools to detect objects in images, count them by class, and export models to TFLite format for mobile deployment.

## Features

- **Object Detection**: Uses YOLOv8 nano model for fast and accurate object detection
- **Object Counting**: Automatically counts detected objects and breaks down counts by class
- **Model Export**: Export YOLOv8 models to TFLite format for mobile and edge devices
- **TFLite Inference**: Run inference using TFLite models for optimized performance on mobile devices
- **Batch Processing**: Process multiple images at once with automated output saving

## Project Structure

```
object-counter-mobile/
├── src/
│   ├── run_yolo_inference.py      # Run YOLO inference on images
│   ├── export_to_tflite.py         # Export YOLOv8 model to TFLite format
│   └── tflite_inference.py         # Run inference using TFLite models
├── data/
│   ├── input/                      # Place input images here
│   └── output/                     # Processed images with detections saved here
├── models/                         # Store exported TFLite models here
├── notebooks/                      # Jupyter notebooks for experimentation
├── yolov8n.pt                      # YOLOv8 nano pretrained weights
└── README.md                       # This file
```

## Requirements

- Python 3.8+
- GPU support (optional, but recommended for faster processing)

### Dependencies

Install required packages:

```bash
pip install ultralytics opencv-python numpy
```

For TFLite inference on mobile/edge devices:
```bash
pip install tflite-runtime
```

Or for full TensorFlow support:
```bash
pip install tensorflow
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Prudhviteja9/object-counter-mobile.git
cd object-counter-mobile
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Run YOLOv8 Inference

Process images in the `data/input/` directory:

```bash
python src/run_yolo_inference.py
```

**What it does:**
- Loads the YOLOv8 nano model
- Processes all `.jpg`, `.jpeg`, and `.png` images in `data/input/`
- Counts total objects and per-class counts
- Saves annotated images to `data/output/`
- Prints detection statistics

**Output example:**
```
Processing data/input/image1.jpg
Total objects detected: 42
Per-class counts: {0: 15, 1: 27}
Saved result to data/output/image1.jpg
```

### 2. Export Model to TFLite

Export the YOLOv8 model for mobile deployment:

```bash
python src/export_to_tflite.py
```

**What it does:**
- Exports the YOLOv8 nano model to TFLite format
- Saves the `.tflite` file in the project root or `models/` directory
- Optimizes the model for mobile and edge devices

**Note:** This process may take a few minutes on first run.

### 3. Run TFLite Inference

Run inference using the exported TFLite model:

```bash
python src/tflite_inference.py
```

**What it does:**
- Automatically finds the `.tflite` model file
- Loads the first image from `data/input/`
- Runs inference and prints output tensor shapes
- Verifies the model works correctly

## Model Information

- **Model**: YOLOv8 Nano (yolov8n)
- **Input Size**: 640x640 pixels
- **Output**: Bounding boxes, class predictions, and confidence scores
- **Classes**: 80 COCO dataset classes (people, cars, animals, etc.)

### Model Download

The YOLOv8 model (`yolov8n.pt`) is automatically downloaded by the ultralytics library on first use. You can also manually download it from the [YOLOv8 releases page](https://github.com/ultralytics/yolov8).

## TFLite Model Optimization

When exporting to TFLite, the model is optimized for:
- **Smaller file size**: ~20% reduction compared to original model
- **Faster inference**: Optimized for mobile CPUs and edge devices
- **Quantization support**: Can be further optimized with post-training quantization

## Examples

### Example 1: Detect objects in a single image
```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
results = model("path/to/image.jpg")
print(f"Objects detected: {len(results[0].boxes)}")
```

### Example 2: Count specific object classes
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("image.jpg")[0]

# Get class names
class_names = results.names

# Count by class
for box in results.boxes:
    class_id = int(box.cls[0])
    class_name = class_names[class_id]
    confidence = box.conf[0]
    print(f"{class_name}: {confidence:.2f}")
```

## Performance

- **YOLOv8 Nano**: ~300 FPS on GPU, ~50 FPS on CPU (640x640 images)
- **TFLite Model**: ~200+ FPS on modern smartphones
- **File Size**: ~6 MB (original), ~4 MB (TFLite)

## Troubleshooting

### Issue: No images processed
- **Solution**: Ensure images are placed in `data/input/` with `.jpg`, `.jpeg`, or `.png` extensions

### Issue: CUDA/GPU not detected
- **Solution**: Install CUDA toolkit and cuDNN compatible with your PyTorch version, or use CPU-only version

### Issue: TFLite model not found
- **Solution**: Run `python src/export_to_tflite.py` first to generate the `.tflite` file

### Issue: Out of memory errors
- **Solution**: Process images in smaller batches or reduce input size

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is open source and available under the MIT License.

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [COCO Dataset](https://cocodataset.org/)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [OpenCV Documentation](https://docs.opencv.org/)

## Contact

For questions or support, please open an issue on GitHub or contact the repository maintainer.

---

**Happy Object Counting!** 🎯
