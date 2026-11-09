from pathlib import Path
import time

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "pretrained" / "yolo11n.pt"
VIDEO_PATH = ROOT / "data" / "samples" / "video.mp4"
OUTPUT_DIR = ROOT / "results" / "video_test"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelul nu exista: {MODEL_PATH}")

    if not VIDEO_PATH.exists():
            raise FileNotFoundError(f"Videoul nu exista: {VIDEO_PATH}")


    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
         raise RuntimeError("Nu merge sa deschida videoul")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))

    if input_fps <= 0:
         input_fps = 30.0

    output_path = OUTPUT_DIR / "detected_video.mp4"

    writer = cv2.VideoWriter(
         str(output_path),
         cv2.VideoWriter_fourcc(*"mp4v"),
         input_fps,
         (width, height),
    )

    if not writer.isOpened():
         cap.release()
         raise RuntimeError("Nu s-a putut crea videoul de iesire")

    frame_count = 0
    total_processing_time = 0.0

    try:
         while True:
            success, frame = cap.read()

            if not success:
                break

            start = time.perf_counter()

            results = model.predict(
                 source=frame,
                 device=device,
                 verbose=False
            )

            annotated = results[0].plot()

            if torch.cuda.is_available():
                 torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            total_processing_time += elapsed

            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            cv2.putText(
                 annotated,
                 f"FPS:{fps:.1f}",
                 (20, 40),
                 cv2.FONT_HERSHEY_SIMPLEX,
                 1,
                 (0, 255, 0),
                 2,
            )

            writer.write(annotated)
            frame_count += 1

            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
    finally:
        cap.release()
        writer.release()

    avg_fps = (
         frame_count / total_processing_time
         if total_processing_time > 0
         else 0.0
    )

    print(f"\nFrames processed: {frame_count}")
    print(f"Average processing FPS: {avg_fps:.2f}")
    print(f"Saved video: {output_path}")

if __name__ == "__main__":
    main()
