#!/usr/bin/env python3
"""Export the PLKSR-Rep x4 model to ONNX and validate the result."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.inference_ntire import load_model


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "model_zoo/19_PLKSRRep_IQAv2Short/net_g_1000.pth"
DEFAULT_OUTPUT = ROOT / "exports/plksr_rep_x4.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--height", type=int, default=64, help="Example LR input height")
    parser.add_argument("--width", type=int, default=64, help="Example LR input width")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export dynamic LR height/width (output dimensions remain 4x)",
    )
    parser.add_argument(
        "--skip-runtime-check",
        action="store_true",
        help="Skip ONNX Runtime numerical comparison",
    )
    return parser.parse_args()


def require_positive_shape(height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError(f"Input dimensions must be positive, got {height}x{width}")


def main() -> None:
    args = parse_args()
    require_positive_shape(args.height, args.width)
    weights = args.weights.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX is required: python -m pip install onnx") from exc

    model_args = SimpleNamespace(arch="plksr_rep", dim=64, n_blocks=12)
    device = torch.device("cpu")
    model = load_model(str(weights), device, model_args).eval()
    example = torch.rand(1, 3, args.height, args.width, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input": {2: "lr_height", 3: "lr_width"},
            "output": {2: "sr_height", 3: "sr_width"},
        }

    with torch.inference_mode():
        reference = model(example).cpu().numpy()
        torch.onnx.export(
            model,
            example,
            str(output),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed: {output}")
    print(f"Input:  NCHW = (1, 3, {args.height}, {args.width})")
    print(f"Output: NCHW = {reference.shape}")

    if not args.skip_runtime_check:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime is required for numerical validation: "
                "python -m pip install onnxruntime"
            ) from exc
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        actual = session.run(["output"], {"input": example.cpu().numpy()})[0]
        max_abs_error = float(np.max(np.abs(reference - actual)))
        np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-5)
        print(f"ONNX Runtime comparison passed; max_abs_error={max_abs_error:.8f}")


if __name__ == "__main__":
    main()
