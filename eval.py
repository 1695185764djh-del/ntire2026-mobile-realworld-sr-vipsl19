import argparse
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    files = sorted(inp.glob("*.png"))
    if not files:
        print("No input PNG found")
        return 2

    bad = []
    for lr in files:
        sr = out / lr.name
        if not sr.exists():
            bad.append(f"missing: {lr.name}")
            continue
        a = cv2.imread(str(lr), cv2.IMREAD_COLOR)
        b = cv2.imread(str(sr), cv2.IMREAD_COLOR)
        if a is None or b is None:
            bad.append(f"unreadable: {lr.name}")
            continue
        eh, ew = a.shape[0] * args.scale, a.shape[1] * args.scale
        if (b.shape[0], b.shape[1]) != (eh, ew):
            bad.append(f"size mismatch: {lr.name}")

    print(f"input_count={len(files)}")
    print(f"bad_count={len(bad)}")
    for x in bad[:20]:
        print(x)
    return 0 if not bad else 3


if __name__ == "__main__":
    raise SystemExit(main())
