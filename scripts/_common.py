"""Shared command-line setup for the project scripts."""
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference.yolo26 import prepare, export, build, run, compare, paths, SETTINGS


def arguments(description, backend=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--task', choices=['detect', 'segment'], default='detect',
                        help='detect: YOLO26n; segment: YOLO26n-seg')
    if backend:
        parser.add_argument('--backend', choices=['pytorch', 'onnx', 'fp32', 'fp16'], default='pytorch')
    return parser
