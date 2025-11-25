"""
Load a TFLite model and run inference on a single image from data/input.
Print output tensor shapes to confirm inference works.

Requirements:
- numpy
- opencv-python
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input"


def load_image(path, size=(640, 640)):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def main():
    tflite_model = None
    # find a .tflite file in project root or models/
    for p in [ROOT, ROOT / "models"]:
        for f in p.glob("*.tflite"):
            tflite_model = f
            break
        if tflite_model:
            break

    if tflite_model is None:
        print("No .tflite model found in project root or models/. Run export_to_tflite.py first.")
        return

    interpreter = tflite.Interpreter(model_path=str(tflite_model))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input details:")
    for d in input_details:
        print(d['shape'], d['dtype'])

    print("Output details:")
    for d in output_details:
        print(d['shape'], d['dtype'])

    # pick first image in data/input
    imgs = list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.png"))
    if not imgs:
        print(f"No images found in {INPUT_DIR} to run inference on.")
        return

    img = load_image(imgs[0])
    # if interpreter expects different dtype, cast
    input_dtype = input_details[0]['dtype']
    img = img.astype(input_dtype)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    outputs = [interpreter.get_tensor(o['index']) for o in output_details]
    for i, out in enumerate(outputs):
        print(f"Output[{i}] shape: {out.shape}, dtype: {out.dtype}")


if __name__ == '__main__':
    main()
