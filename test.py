import argparse
import os
import subprocess
import sys


MODEL_PATH = "model_zoo/19_PLKSRRep_IQAv2Short/net_g_1000.pth"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--prepad", type=int, default=16)
    parser.add_argument("--fp16", type=int, default=0)
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.output, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(root, "scripts", "inference_ntire.py"),
        "--weights",
        os.path.join(root, args.model_path),
        "--input",
        args.input,
        "--output",
        args.output,
        "--arch",
        "plksr_rep",
        "--dim",
        "64",
        "--n_blocks",
        "12",
        "--prepad",
        str(args.prepad),
    ]
    if args.fp16 == 1:
        cmd.append("--fp16")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = root
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
