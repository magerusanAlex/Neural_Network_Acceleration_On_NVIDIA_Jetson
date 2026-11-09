
from pathlib import Path
from collections import defaultdict
import re
import time

import cv2
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

FRAMES_DIR = ROOT / "data/raw/flir_adas_v2/video_rgb_test/data"
MODEL_PATH = ROOT / "models/pretrained/yolo11n.pt"
OUTPUT_DIR = ROOT / "results/sequence_inference"

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

    # Pentru demo alegem cea mai lunga secventa.
    sequence_id, frames = max(
        sequences.items(),
        key=lambda item: len(item[1])
    )

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

    output_path = OUTPUT_DIR / "flir_rgb_detection.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_FPS,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("Nu pot crea fisierul MP4.")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = YOLO(str(MODEL_PATH))

    total_inference_time = 0.0
    processed_frames = 0

    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))

            if frame is None:
                print(f"Skipping: {frame_path.name}")
                continue

            start = time.perf_counter()

            results = model.predict(
                source=frame,
                device=device,
                verbose=False
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            total_inference_time += elapsed

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

    average_fps = (
        processed_frames / total_inference_time
        if total_inference_time > 0 else 0
    )

    print(f"\nProcessed frames: {processed_frames}")
    print(f"Average predict FPS: {average_fps:.2f}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
