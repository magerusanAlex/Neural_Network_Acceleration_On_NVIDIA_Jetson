"""Run all four formats on the reference image and compare boxes and masks."""
from _common import arguments, run, compare


def main():
    args = arguments(__doc__).parse_args()
    for backend in ['pytorch', 'onnx', 'fp32', 'fp16']:
        run(args.task, backend, count=0)
        if backend != 'pytorch':
            compare(args.task, backend, image_only=True)


if __name__ == '__main__':
    main()
