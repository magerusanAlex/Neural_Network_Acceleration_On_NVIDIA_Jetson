"""Reproducible CPU smoke validation of PyTorch versus exported ONNX."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def compare(a, b):
    # Greedy, same-class matching in descending PyTorch confidence order.
    used = set()
    pairs = []
    for x in sorted(a, key=lambda row: -row[4]):
        candidates = []
        for j, y in enumerate(b):
            if j in used or x[5] != y[5]:
                continue
            intersection = np.prod(np.maximum(0, np.minimum(x[2:4], y[2:4]) - np.maximum(x[:2], y[:2])))
            union = np.prod(x[2:4] - x[:2]) + np.prod(y[2:4] - y[:2]) - intersection
            iou = float(intersection / union) if union > 0 else 0.0
            candidates.append((iou, j))
        if candidates:
            iou, j = max(candidates)
            if iou >= 0.5:
                used.add(j)
                pairs.append({"iou": iou, "box_delta_px": float(np.max(np.abs(x[:4] - b[j, :4]))),
                              "score_delta": float(abs(x[4] - b[j, 4]))})
    unmatched_pt = len(a) - len(pairs)
    unmatched_onnx = len(b) - len(pairs)
    passed = (unmatched_pt == unmatched_onnx == 0 and all(
        p["iou"] >= 0.99 and p["box_delta_px"] <= 1.0 and p["score_delta"] <= 0.001 for p in pairs))
    return {"pytorch_count": len(a), "onnx_count": len(b), "matched": len(pairs),
            "unmatched_pytorch": unmatched_pt, "unmatched_onnx": unmatched_onnx,
            "min_iou": min((p["iou"] for p in pairs), default=None),
            "max_box_delta_px": max((p["box_delta_px"] for p in pairs), default=None),
            "max_score_delta": max((p["score_delta"] for p in pairs), default=None), "passed": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "results/onnx_validation/report.json")
    args = parser.parse_args()
    folder = ROOT / "data/raw/flir_adas_v2/images_rgb_val/data"
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not 1 <= args.count <= len(files):
        parser.error(f"--count must be between 1 and {len(files)}")
    selected = [files[i] for i in np.linspace(0, len(files) - 1, args.count, dtype=int)]
    paths = {"pytorch": ROOT / "models/pretrained/yolo11n.pt", "onnx": ROOT / "models/onnx/yolo11n.onnx"}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    onnx.checker.check_model(onnx.load(str(paths["onnx"])))
    pt = YOLO(str(paths["pytorch"]))
    ox = YOLO(str(paths["onnx"]), task="detect")
    settings = dict(imgsz=640, rect=False, conf=0.25, iou=0.7, max_det=300,
                    device="cpu", verbose=False, save=False, augment=False)
    records = []
    for path in selected:
        a = pt.predict(source=str(path), **settings)[0]
        b = ox.predict(source=str(path), **settings)[0]
        if a.names != b.names:
            raise RuntimeError("Class mappings differ between models")
        result = compare(a.boxes.data.cpu().numpy(), b.boxes.data.cpu().numpy())
        result["image"] = path.relative_to(ROOT).as_posix()
        result["image_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(result)
        print(f"{'PASS' if result['passed'] else 'REVIEW'} {path.name}: PT={result['pytorch_count']}, ONNX={result['onnx_count']}, matched={result['matched']}", flush=True)
    # Empty results are valid comparisons, but cannot alone validate the export.
    passed = all(r["passed"] for r in records) and sum(r["matched"] for r in records) > 0
    report = {"passed": passed, "scope": "CPU detection consistency smoke test; not accuracy or speed benchmark",
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           "ultralytics": ultralytics.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
              "model_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()},
              "settings": settings, "matching": "greedy same-class, descending PyTorch confidence, IoU >= 0.5",
              "tolerances": {"min_iou": 0.99, "max_box_delta_px": 1.0, "max_score_delta": 0.001},
              "images": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {args.output}\nOverall: {'PASS' if passed else 'REVIEW'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
