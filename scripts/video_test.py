"""Run YOLO26 on an existing video file (use sequence_inference for FLIR JPGs)."""
from pathlib import Path
import math
import cv2
from ultralytics import YOLO
from _common import arguments, paths, SETTINGS, ROOT


def main():
    parser = arguments(__doc__, backend=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    weight = paths(args.task)[args.backend]
    if not weight.is_file():
        raise FileNotFoundError(weight)
    model = YOLO(str(weight), task=args.task)
    cap = cv2.VideoCapture(str(args.source))
    writer = None
    count = 0
    output = ROOT / 'results/yolo26' / args.task / args.backend / 'video' / (args.source.stem + '_annotated.mp4')
    if output.resolve() == args.source.resolve():
        cap.release()
        raise ValueError('Input and output must differ')
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open video')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 20.0
            print('Source FPS unavailable; using 20 FPS playback')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError('Cannot create output video')
            result = model.predict(source=frame, device='cpu' if args.backend == 'onnx' else 0,
                                   data='coco.yaml', verbose=False, save=False, **SETTINGS)[0]
            writer.write(result.plot())
            count += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if count == 0:
        raise RuntimeError('No decodable frames')
    print(f'Processed {count} frames: {output}')


if __name__ == '__main__':
    main()
