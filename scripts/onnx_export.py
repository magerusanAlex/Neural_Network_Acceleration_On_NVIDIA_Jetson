
from pathlib import Path
import shutil

import onnx
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

# Alegem "detect" pentru YOLO26n sau "segment" pentru YOLO26n-seg.
TASK = "detect"
MODEL_NAME = "yolo26n" if TASK == "detect" else "yolo26n-seg"
MODEL_DIR = ROOT / "models/yolo26" / TASK


MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}.pt"
OUTPUT_DIR = MODEL_DIR


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME}...")
    model = YOLO(str(MODEL_PATH), task=TASK)

    print("Exporting to ONNX...")
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=640,
            batch=1,
            dynamic=False,
            nms=None,  # Pastram NMS in afara modelului, ca la predictie.
            device="cpu",
        )
    )

    output_path = OUTPUT_DIR / f"{MODEL_NAME}.onnx"

    if exported_path.resolve() != output_path.resolve():
        shutil.copy2(exported_path, output_path)

    print("Checking ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    print(f"ONNX export successful: {output_path}")

    for tensor in onnx_model.graph.input:
        print(f"Input: {tensor.name}")

    for tensor in onnx_model.graph.output:
        print(f"Output: {tensor.name}")


if __name__ == "__main__":
    main()
