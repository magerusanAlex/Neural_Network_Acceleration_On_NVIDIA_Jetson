"""Shared YOLO26 detection/instance-segmentation experiments, with external NMS."""
from pathlib import Path
import json
import hashlib
import time
import platform

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT / 'data/raw/flir_adas_v2/images_rgb_val/data/video-Chrih8BX8fbZLmsoX-frame-001292-DxauLXa3S44rh37Z7.jpg'
FRAMES = ROOT / 'data/raw/flir_adas_v2/video_rgb_test/data'
SEQUENCE = 'video-RMxN6a4CcCeLGu4tA'
SETTINGS = dict(imgsz=640, rect=False, conf=0.25, iou=0.7, max_det=300, nms=None, augment=False)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def paths(task):
    name = 'yolo26n' if task == 'detect' else 'yolo26n-seg'
    folder = ROOT / 'models/yolo26' / task
    return {k: folder / (name + ext) for k, ext in {
        'pytorch': '.pt', 'onnx': '.onnx', 'fp32': '_fp32.engine', 'fp16': '_fp16.engine'}.items()}


def prepare(task):
    p = paths(task)['pytorch']
    p.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(p), task=task)
    if not p.is_file():
        raise RuntimeError(f'Weights were not saved at {p}')
    print(f'READY {p}, classes={len(model.names)}', flush=True)


def export(task):
    import onnx
    p = paths(task)
    model = YOLO(str(p['pytorch']), task=task)
    output = Path(model.export(format='onnx', imgsz=640, batch=1, dynamic=False,
                              nms=None, opset=18, simplify=False, device='cpu'))
    if output.resolve() != p['onnx'].resolve():
        raise RuntimeError(f'Unexpected export location: {output}')
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    print('ONNX CHECK OK', [(t.name, [d.dim_value for d in t.type.tensor_type.shape.dim]) for t in graph.graph.output], flush=True)


def build(task, precision):
    import tensorrt as trt
    p = paths(task)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(p['onnx'])):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if network.num_inputs != 1 or tuple(network.get_input(0).shape) != (1, 3, 640, 640):
        raise RuntimeError('Expected static input [1, 3, 640, 640]')
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.clear_flag(trt.BuilderFlag.INT8)
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        config.clear_flag(trt.BuilderFlag.FP16)
    print(f'BUILDING {task} {precision}', flush=True)
    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT build failed')
    elapsed = time.perf_counter() - start
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError('TensorRT deserialization failed')
    io = [{"name": engine.get_tensor_name(i), "shape": list(engine.get_tensor_shape(engine.get_tensor_name(i))),
           "dtype": str(engine.get_tensor_dtype(engine.get_tensor_name(i))),
           "mode": str(engine.get_tensor_mode(engine.get_tensor_name(i)))} for i in range(engine.num_io_tensors)]
    p[precision].write_bytes(bytes(serialized))
    write_json(p[precision].with_suffix('.build.json'), dict(task=task, precision_enabled=precision,
        tensorrt=trt.__version__, gpu=torch.cuda.get_device_name(0), platform=platform.platform(),
        onnx_sha256=hashlib.sha256(p['onnx'].read_bytes()).hexdigest(), build_seconds=elapsed,
        workspace_bytes=1 << 30, io=io))
    print(f'BUILT {p[precision]}: {elapsed:.1f}s, {p[precision].stat().st_size / 2**20:.2f} MiB, IO={io}', flush=True)


def run(task, backend, count):
    p = paths(task)[backend]
    if not p.is_file():
        raise FileNotFoundError(p)
    selected = sorted(FRAMES.glob(f'{SEQUENCE}-frame-*.jpg'), key=lambda x: int(x.name.split('-frame-')[1].split('-')[0]))[:count]
    if len(selected) != count:
        raise RuntimeError(f'Requested {count} frames, found {len(selected)}')
    out = ROOT / 'results/yolo26' / task / backend
    out.mkdir(parents=True, exist_ok=True)
    device = 'cpu' if backend == 'onnx' else 0
    model = YOLO(str(p), task=task)
    rows = []
    writer = None
    try:
        for index, source in enumerate([IMAGE] + selected):
            frame = cv2.imread(str(source))
            if frame is None:
                raise RuntimeError(f'Cannot read {source}')
            result = model.predict(source=frame, device=device, data='coco.yaml', verbose=False,
                                   save=False, **SETTINGS)[0]
            boxes = result.boxes.data.cpu().numpy()
            row = dict(image=source.relative_to(ROOT).as_posix(), boxes=boxes.tolist())
            if task == 'segment':
                if len(boxes) and (result.masks is None or len(result.masks.data) != len(boxes)):
                    raise RuntimeError('Masks and boxes do not correspond')
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy().astype(bool)
                    mask_file = f'masks_{index:04d}.npz'
                    np.savez_compressed(out / mask_file, shape=np.array(masks.shape), bits=np.packbits(masks.reshape(-1)))
                    row['masks'] = mask_file
            annotated = result.plot()
            if index == 0:
                if not cv2.imwrite(str(out / 'image.jpg'), annotated):
                    raise RuntimeError('Cannot save image')
            else:
                if writer is None:
                    writer = cv2.VideoWriter(str(out / 'sequence.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20.0,
                                             (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('Cannot create video')
                writer.write(annotated)
            rows.append(row)
            if index % 25 == 0:
                print(f'RUN {task} {backend}: image + {index}/{count} frames', flush=True)
    finally:
        if writer is not None:
            writer.release()
    report = dict(task=task, backend=backend, device=device, sequence=SEQUENCE, video_frames=count,
                  settings=SETTINGS, classes=model.predictor.model.names, model_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                  versions=dict(python=platform.python_version(), ultralytics=ultralytics.__version__, torch=torch.__version__),
                  scope='Prediction consistency only; not ground-truth accuracy or speed benchmark', frames=rows)
    write_json(out / ('predictions.json' if count else 'image_predictions.json'), report)
    print(f'DONE {task} {backend}: image={len(rows[0]["boxes"])}, sequence={sum(len(x["boxes"]) for x in rows[1:])}', flush=True)


def box_iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = lambda x: max(0, x[2]-x[0])*max(0, x[3]-x[1])
    union = area(a)+area(b)-inter
    return inter/union if union else 0.0


def masks_at(folder, row):
    if 'masks' not in row:
        return None
    with np.load(folder / row['masks']) as data:
        shape = tuple(data['shape'])
        return np.unpackbits(data['bits'], count=int(np.prod(shape))).reshape(shape).astype(bool)


def compare(task, backend, image_only=False):
    base = ROOT / 'results/yolo26' / task
    filename = 'image_predictions.json' if image_only else 'predictions.json'
    a = json.loads((base / 'pytorch' / filename).read_text())
    b = json.loads((base / backend / filename).read_text())
    if a['settings'] != b['settings'] or a['classes'] != b['classes'] or [r['image'] for r in a['frames']] != [r['image'] for r in b['frames']]:
        raise RuntimeError('Inconsistent frames, settings or class maps')
    summaries = []
    for section, aa, bb in [('image', a['frames'][:1], b['frames'][:1]), ('sequence', a['frames'][1:], b['frames'][1:])]:
        pairs=[]; unmatched_a=0; unmatched_b=0; different=[]; mask_ious=[]
        for x,y in zip(aa,bb):
            used=set(); matched=0
            ma=masks_at(base/'pytorch',x); mb=masks_at(base/backend,y)
            for i in sorted(range(len(x['boxes'])), key=lambda i:-x['boxes'][i][4]):
                box=x['boxes'][i]
                candidates=[(box_iou(box, other),j) for j,other in enumerate(y['boxes']) if j not in used and box[5]==other[5]]
                overlap,j=max(candidates) if candidates else (0,None)
                if overlap < 0.5:
                    continue
                used.add(j);matched+=1;other=y['boxes'][j]
                pairs.append((overlap,max(abs(u-v) for u,v in zip(box[:4],other[:4])),abs(box[4]-other[4])))
                if task=='segment':
                    if ma is None or mb is None or ma.shape[1:]!=mb.shape[1:]:
                        raise RuntimeError('Missing or incompatible masks')
                    union=np.count_nonzero(ma[i]|mb[j]); intersection=np.count_nonzero(ma[i]&mb[j])
                    mask_ious.append(float(intersection/union) if union else 1.0)
            unmatched_a+=len(x['boxes'])-matched;unmatched_b+=len(y['boxes'])-matched
            if matched!=len(x['boxes']) or matched!=len(y['boxes']):different.append(x['image'])
        summaries.append(dict(section=section, matched=len(pairs), unmatched_pytorch=unmatched_a, unmatched_candidate=unmatched_b,
            min_box_iou=min((v[0] for v in pairs),default=None), max_coordinate_delta_px=max((v[1] for v in pairs),default=None),
            max_score_delta=max((v[2] for v in pairs),default=None), min_mask_iou=min(mask_ious,default=None),
            mean_mask_iou=float(np.mean(mask_ious)) if mask_ious else None, different_frames=different))
    report=dict(task=task, candidate=backend, reference='pytorch', matching='greedy same-class IoU >= 0.5, descending reference confidence',
        scope='Agreement, not ground-truth accuracy. No automatic pass/fail tolerance.', summaries=summaries)
    if image_only:
        report['summaries'] = report['summaries'][:1]
    write_json(base / backend / ('image_comparison.json' if image_only else 'comparison.json'), report)
    print(json.dumps(report,indent=2), flush=True)

