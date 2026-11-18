"""Build and reload a YOLO26 TensorRT engine on the current GPU."""
from _common import arguments, build


def main():
    parser = arguments(__doc__)
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    args = parser.parse_args()
    build(args.task, args.precision)


if __name__ == '__main__':
    main()
