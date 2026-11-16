
from pathlib import Path
import shutil

import onnx
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models/pretrained/yolo11n.pt"
OUTPUT_DIR = ROOT / "models/onnx"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO11n...")
    model = YOLO(str(MODEL_PATH))

    print("Exporting to ONNX...")
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=640,
            batch=1,
            dynamic=False,
            nms=False,
            device="cpu",
        )
    )

    output_path = OUTPUT_DIR / "yolo11n.onnx"

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
