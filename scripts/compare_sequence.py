"""Compare saved sequence predictions against PyTorch; no inference is rerun."""
from _common import arguments, compare


def main():
    parser = arguments(__doc__)
    parser.add_argument('--backend', choices=['onnx', 'fp32', 'fp16'], required=True)
    args = parser.parse_args()
    compare(args.task, args.backend)


if __name__ == '__main__':
    main()
