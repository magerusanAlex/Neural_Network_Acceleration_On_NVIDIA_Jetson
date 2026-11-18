from pathlib import Path
import time

import tensorrt as trt

ROOT = Path(__file__).resolve().parents[1]

# Alegem "detect" pentru YOLO26n sau "segment" pentru YOLO26n-seg.
TASK = "detect"
MODEL_NAME = "yolo26n" if TASK == "detect" else "yolo26n-seg"
MODEL_DIR = ROOT / "models/yolo26" / TASK


ONNX_PATH = MODEL_DIR / f"{MODEL_NAME}.onnx"

PRECISION = "fp16"
ENGINE_PATH = MODEL_DIR / f"{MODEL_NAME}_{PRECISION}.engine"

WORKSPACE_BYTES = 1024*1024*1024

def main():

    if PRECISION not in {"fp32", "fp16"}:
        raise ValueError("PRECISION trebuie să fie 'fp32' sau 'fp16'.")

    if not ONNX_PATH.is_file():
        raise FileNotFoundError(ONNX_PATH)

    print(f"TensorRT ver. = {trt.__version__}")

    logger = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(ONNX_PATH)):
        for index in range(parser.num_errors):
            print(parser.get_error(index))

        raise RuntimeError("TensorRT nu a putut interpreta modelul ONNX.")

    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        print(
            f"Intrare: {tensor.name}, "
            f"formă={tuple(tensor.shape)}, tip={tensor.dtype}"
        )

    for index in range(network.num_outputs):
        tensor = network.get_output(index)
        print(
            f"Ieșire: {tensor.name}, "
            f"formă={tuple(tensor.shape)}, tip={tensor.dtype}"
        )

    if network.num_inputs != 1:
        raise RuntimeError("eroare")

    input_shape = tuple(network.get_input(0).shape)

    if input_shape != (1, 3, 640, 640):
        raise RuntimeError(f"{input_shape}")

    config = builder.create_builder_config()

    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        WORKSPACE_BYTES,
    )

    #config.clear_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.INT8)
    config.clear_flag(trt.BuilderFlag.TF32)

    if PRECISION == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        config.clear_flag(trt.BuilderFlag.FP16)

    start = time.perf_counter()

    serialized_engine = builder.build_serialized_network(
        network,
        config,
    )

    elapsed = time.perf_counter() - start

    if serialized_engine is None:
        raise RuntimeError(
            "Construirea motorului a eșuat. Verifică mesajele TensorRT."
        )

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    if engine is None:
        raise RuntimeError("Motorul construit nu a putut fi încărcat.")

    ENGINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENGINE_PATH.write_bytes(bytes(serialized_engine))

    print(f"\nMotor salvat: {ENGINE_PATH}")
    print(f"Dimensiune: {ENGINE_PATH.stat().st_size / (1024 ** 2):.2f} MiB")
    print(f"Timp de construire: {elapsed:.2f} secunde")
    print("Verificare încărcare motor: OK")

if __name__ == "__main__":
    main()
