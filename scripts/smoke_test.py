from pathlib import Path
import time

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pretrained" / "yolo11n.pt"
IMAGE_PATH = ROOT / "data" / "samples" / "bus.jpg"
OUTPUT_DIR = ROOT / "results" / "smoke_test"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("SSC P3 - YOLO Smoke Test")
    print("=" * 50)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    device = 0 if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    # Here we load the pretrained YOLO model
    model = YOLO(str(MODEL_PATH))

    # Warmup
    for _ in range(3):
        model.predict(
            source=str(IMAGE_PATH),
            device=device,
            verbose=False
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Timed inference
    start = time.perf_counter()

    results = model.predict(
        source=str(IMAGE_PATH),
        device=device,
        verbose=False
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000

    result = results[0]

    print(f"\nTotal predict call: {elapsed_ms:.2f} ms")
    print(f"Ultralytics timing: {result.speed}")

    print("\nDetected objects:")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]

        print(f"  {class_name}: {confidence:.2f}")

    annotated_image = result.plot()

    output_path = OUTPUT_DIR / "bus_detected.jpg"

    success = cv2.imwrite(str(output_path), annotated_image)

    if not success:
        raise RuntimeError("Could not save output image")

    print(f"\nSaved result: {output_path}")
    print("Smoke test completed successfully!")


if __name__ == "__main__":
    main()
