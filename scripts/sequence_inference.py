from pathlib import Path
from collections import defaultdict
import re
import time

import json
import cv2
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

# Alegem "detect" pentru YOLO26n sau "segment" pentru YOLO26n-seg.
TASK = "detect"
MODEL_NAME = "yolo26n" if TASK == "detect" else "yolo26n-seg"
MODEL_DIR = ROOT / "models/yolo26" / TASK


BACKEND ="tensorrt_fp16" #determina ce fisier incarcam si cum numim rezultatele

FRAMES_DIR = ROOT / "data/raw/flir_adas_v2/video_rgb_test/data"

MODEL_PATH = {
    "pytorch": MODEL_DIR / f"{MODEL_NAME}.pt",
    "onnx": MODEL_DIR / f"{MODEL_NAME}.onnx",
    "tensorrt_fp32": MODEL_DIR / f"{MODEL_NAME}_fp32.engine",
    "tensorrt_fp16": MODEL_DIR / f"{MODEL_NAME}_fp16.engine"
}

MODEL_PATH = MODEL_PATH[BACKEND]

SEQUENCE_ID = "video-RMxN6a4CcCeLGu4tA"

OUTPUT_DIR = ROOT / "results/yolo26" / TASK / "sequence_inference"

# Limitam primul test la 150 de cadre.
MAX_FRAMES = 150

# FPS de redare pentru videoclipul rezultat.
# Nu presupunem ca acesta este FPS-ul original.
OUTPUT_FPS = 20.0


def extract_sequence_info(path):
    """
    Extrage identificatorul videoclipului si numarul cadrului
    din nume de forma:
    video-<id>-frame-000889-<id>.jpg
    """
    match = re.match(
        r"^(video-.+?)-frame-(\d+)-",
        path.stem
    )

    if not match:
        return None

    return match.group(1), int(match.group(2))


def find_sequence_frames(folder):
    sequences = defaultdict(list)

    for path in folder.iterdir():
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        info = extract_sequence_info(path)

        if info is None:
            continue

        sequence_id, frame_number = info
        sequences[sequence_id].append((frame_number, path))

    if not sequences:
        raise RuntimeError(
            "Nu am gasit cadre cu numele asteptat."
        )

    if SEQUENCE_ID not in sequences:
        raise RuntimeError(f"Nu am găsit secvența: {SEQUENCE_ID}")

    sequence_id = SEQUENCE_ID
    frames = sequences[sequence_id]

    frames.sort(key=lambda item: item[0])

    return sequence_id, [path for _, path in frames]


def main():
    if not FRAMES_DIR.exists():
        raise FileNotFoundError(FRAMES_DIR)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sequence_id, frames = find_sequence_frames(FRAMES_DIR)
    frames = frames[:MAX_FRAMES]

    print(f"Sequence: {sequence_id}")
    print(f"Selected frames: {len(frames)}")

    first_frame = cv2.imread(str(frames[0]))

    if first_frame is None:
        raise RuntimeError("Nu pot citi primul cadru.")

    height, width = first_frame.shape[:2]

    output_path = OUTPUT_DIR / f"flir_rgb_{BACKEND}.mp4"
    predictions_path = OUTPUT_DIR / f"flir_rgb_{BACKEND}.json"
    predictions = []

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_FPS,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("Nu pot crea fisierul MP4.")

    device = "cpu" if BACKEND == "onnx" else 0
    print(f"Backend: {BACKEND}")
    print(f"Device: {device}")

    model = YOLO(str(MODEL_PATH), task=TASK)

    total_inference_time = 0.0
    processed_frames = 0

    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))

            if frame is None:
                raise RuntimeError(f"Nu pot citi cadrul: {frame_path.name}")

            if device != "cpu":
                torch.cuda.synchronize(device)

            start = time.perf_counter()

            results = model.predict(
                source=frame,
                imgsz=640,
                rect=False,
                conf=0.25,
                iou=0.7,
                max_det=300,
                device=device,
                data="coco.yaml",
                nms=None,
                verbose=False,
                save=False,
            )

            if device != "cpu":
                torch.cuda.synchronize(device)

            elapsed = time.perf_counter() - start
            total_inference_time += elapsed

            result = results[0]

            detections = []

            for box in result.boxes:
                class_id = int(box.cls.item())

                detections.append({
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": float(box.conf.item()),
                    "xyxy": box.xyxy[0].cpu().tolist(),
                })

            predictions.append({
                "frame": frame_path.name,
                "detections": detections,
            })

            annotated = results[0].plot()

            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            cv2.putText(
                annotated,
                f"Processing FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            writer.write(annotated)
            processed_frames += 1

            if processed_frames % 25 == 0:
                print(f"Processed {processed_frames} frames")

    finally:
        writer.release()

    report = {
        "backend": BACKEND,
        "sequence": sequence_id,
        "device": device,
        "settings": {
            "imgsz": 640,
            "rect": False,
            "conf": 0.25,
            "iou": 0.7,
            "max_det": 300,
        },
        "processed_frames": processed_frames,
        "frames": predictions,
    }

    predictions_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Predicții salvate: {predictions_path}")

    average_fps = (
        processed_frames / total_inference_time
        if total_inference_time > 0 else 0
    )

    print(f"\nProcessed frames: {processed_frames}")
    print(f"Average predict FPS: {average_fps:.2f}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
