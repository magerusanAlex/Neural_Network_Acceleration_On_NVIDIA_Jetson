"""Download/load the selected YOLO26 model and test the reference FLIR image."""
from _common import arguments, prepare, run


def main():
    args = arguments(__doc__).parse_args()
    prepare(args.task)
    run(args.task, 'pytorch', count=0)


if __name__ == '__main__':
    main()
