#!/usr/bin/env python3
"""Export the PLKSR-Rep x4 model to a Core ML ML Program package."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from scripts.inference_ntire import load_model


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "model_zoo/19_PLKSRRep_IQAv2Short/net_g_1000.pth"
DEFAULT_OUTPUT = ROOT / "exports/plksr_rep_x4.mlpackage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--height", type=int, default=64, help="Default LR input height")
    parser.add_argument("--width", type=int, default=64, help="Default LR input width")
    parser.add_argument(
        "--flexible",
        action="store_true",
        help="Allow LR height/width in the range configured below",
    )
    parser.add_argument("--min-size", type=int, default=16)
    parser.add_argument("--max-size", type=int, default=2048)
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Keep Core ML compute precision at FP32 (default: FP16 weights/compute)",
    )
    parser.add_argument(
        "--tensor-io",
        action="store_true",
        help="Use NCHW MultiArray input/output instead of Xcode-previewable RGB images",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.height <= 0 or args.width <= 0:
        raise ValueError(f"Input dimensions must be positive, got {args.height}x{args.width}")
    if args.flexible:
        if args.min_size <= 0 or args.max_size < args.min_size:
            raise ValueError("Expected 0 < min-size <= max-size")
        if not (args.min_size <= args.height <= args.max_size):
            raise ValueError("Default height must be inside the flexible size range")
        if not (args.min_size <= args.width <= args.max_size):
            raise ValueError("Default width must be inside the flexible size range")


class ImageIOWrapper(nn.Module):
    """Convert the normalized tensor output to an RGB pixel buffer in [0, 255]."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model(image)
        return torch.clamp(output, 0.0, 1.0) * 255.0


def main() -> None:
    args = parse_args()
    validate_args(args)
    weights = args.weights.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    if output.suffix != ".mlpackage":
        raise ValueError("ML Program output path must end with .mlpackage")

    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError("coremltools is required: python -m pip install coremltools") from exc

    model_args = SimpleNamespace(arch="plksr_rep", dim=64, n_blocks=12)
    model = load_model(str(weights), torch.device("cpu"), model_args).eval()
    export_model = model if args.tensor_io else ImageIOWrapper(model).eval()
    example = torch.rand(1, 3, args.height, args.width)
    with torch.inference_mode():
        traced = torch.jit.trace(export_model, example, check_trace=False)
        expected_shape = tuple(export_model(example).shape)

    if args.flexible:
        height = ct.RangeDim(args.min_size, args.max_size, default=args.height)
        width = ct.RangeDim(args.min_size, args.max_size, default=args.width)
        input_shape = ct.Shape(shape=(1, 3, height, width))
    else:
        input_shape = example.shape

    if args.tensor_io:
        coreml_inputs = [ct.TensorType(name="input", shape=input_shape)]
        coreml_outputs = [ct.TensorType(name="output")]
    else:
        coreml_inputs = [
            ct.ImageType(
                name="image",
                shape=input_shape,
                scale=1.0 / 255.0,
                color_layout=ct.colorlayout.RGB,
            )
        ]
        coreml_outputs = [
            ct.ImageType(
                name="super_resolved_image",
                color_layout=ct.colorlayout.RGB,
            )
        ]

    precision = ct.precision.FLOAT32 if args.fp32 else ct.precision.FLOAT16
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=coreml_inputs,
        outputs=coreml_outputs,
        compute_precision=precision,
        minimum_deployment_target=ct.target.iOS15,
    )
    mlmodel.author = "Yin Jian"
    mlmodel.short_description = "PLKSR-Rep 4x mobile real-world image super-resolution"
    if args.tensor_io:
        mlmodel.input_description["input"] = "Normalized RGB tensor in NCHW layout, range [0, 1]"
        mlmodel.output_description["output"] = "4x super-resolved RGB tensor in NCHW layout"
    else:
        mlmodel.input_description["image"] = "Low-resolution RGB image"
        mlmodel.output_description["super_resolved_image"] = "4x super-resolved RGB image"
    mlmodel.user_defined_metadata["scale"] = "4"
    mlmodel.user_defined_metadata["source_checkpoint"] = weights.name
    mlmodel.user_defined_metadata["io_type"] = "multiarray" if args.tensor_io else "image"

    output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output))
    print(f"Core ML package saved: {output}")
    print(f"Default input:  NCHW = {tuple(example.shape)}")
    print(f"Expected output: NCHW = {expected_shape}")
    print(f"Compute precision: {'FP32' if args.fp32 else 'FP16'}")
    print(f"I/O type: {'MultiArray' if args.tensor_io else 'RGB Image (Xcode Preview compatible)'}")
    print("Note: Core ML prediction must be validated on macOS/iOS with Core ML runtime.")


if __name__ == "__main__":
    main()
