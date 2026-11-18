"""Run a YOLO26 format on the reference image and ordered FLIR sequence."""
from _common import arguments, run


def main():
    parser = arguments(__doc__, backend=True)
    parser.add_argument('--count', type=int, default=150)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    run(args.task, args.backend, args.count)


if __name__ == '__main__':
    main()
