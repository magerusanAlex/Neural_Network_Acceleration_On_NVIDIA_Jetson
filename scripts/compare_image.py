from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    ROOT 
    / "data/raw/flir_adas_v2/images_rgb_val/data"
    / "video-Chrih8BX8fbZLmsoX-frame-001292-DxauLXa3S44rh37Z7.jpg"
)

OUTPUT_DIR = ROOT / "results/image_comparison"

MODELS = {
    "pytorch": ROOT / "models/pretrained/yolo11n.pt",
    "onnx": ROOT / "models/onnx/yolo11n.onnx"
}

def main():

    for path in [IMAGE_PATH, *MODELS.values()]:
        if not path.is_file():
            raise FileNotFoundError(f"Nu gasim fisierul {path}")

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError("Nu exista imaginea")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, model_path in MODELS.items():
        model = YOLO(str(model_path), task="detect")

        result = model.predict(
            source=image.copy(),
            imgsz = 640,
            rect = False,
            conf = 0.25,
            iou = 0.7,
            max_det = 300,
            device = "cpu",
            verbose = False,
            save = False,
        )[0]

        print(f"\n--- {model_name.upper()} ---")
        print(f"Număr de detecții: {len(result.boxes)}")
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            coordinates = box.xyxy[0].cpu().tolist()

            print(
                f"{result.names[class_id]:15s} "
                f"scor={confidence:.6f} "
                f"chenar={[round(value, 2) for value in coordinates]}"
            )

        output_path = OUTPUT_DIR / f"{model_name}_detections.jpg"

        if not cv2.imwrite(str(output_path), result.plot()):
            raise RuntimeError(f"Nu am putut salva: {output_path}")

        print(f"Imagine salvată: {output_path}")


if __name__ == "__main__":
    main()