"""Export static YOLO26 ONNX with external NMS and check the graph."""
from _common import arguments, export


def main():
    args = arguments(__doc__).parse_args()
    export(args.task)


if __name__ == '__main__':
    main()
