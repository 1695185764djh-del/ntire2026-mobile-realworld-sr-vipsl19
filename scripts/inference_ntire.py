#!/usr/bin/env python3
# NTIRE Mobile Real-World SR inference (FP16 option)
import argparse
import importlib.util
import os
from glob import glob

import cv2
import torch
import torch.nn.functional as F
import random
import numpy as np

import plksr.archs  # noqa: F401
from basicsr.archs import build_network
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils import get_root_logger


def _clear_arch_registry(*names):
    try:
        from basicsr.utils.registry import ARCH_REGISTRY

        for n in names:
            ARCH_REGISTRY._obj_map.pop(n, None)
    except Exception:
        pass


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_span_model(weights, device, args):
    span_root = args.span_root
    span_arch_py = os.path.join(span_root, "basicsr", "archs", "span_arch.py")
    if not os.path.isfile(span_arch_py):
        raise FileNotFoundError(f"Cannot find SPAN arch file: {span_arch_py}")

    _clear_arch_registry("SPAN", "SPANFlex")
    mod_span = _load_module("span_arch_local", span_arch_py)
    cls = getattr(mod_span, "SPANFlex" if args.arch == "spanflex" else "SPAN")

    kwargs = dict(
        num_in_ch=3,
        num_out_ch=3,
        feature_channels=args.channels,
        upscale=4,
        img_range=255.0,
        rgb_mean=(0.4488, 0.4371, 0.4040),
    )
    if args.arch == "spanflex":
        kwargs.update(block_type=args.block_type, num_blocks=args.num_block)
    else:
        kwargs.update(block_type=args.block_type)

    net_g = cls(**kwargs).to(device).eval()

    ckpt = torch.load(weights, map_location=device)
    params = ckpt.get("params_ema", ckpt.get("params", ckpt))
    if isinstance(params, dict) and params and next(iter(params)).startswith("module."):
        params = {k[len("module."):]: v for k, v in params.items()}
    net_g.load_state_dict(params, strict=True)

    return net_g


def load_model(weights, device, args):
    if args.arch == 'plksr_rep':
        net_g_opt = dict(
            type='PLKSR_Rep',
            dim=args.dim,
            n_blocks=args.n_blocks,
            upscaling_factor=4,
            ccm_type='DCCM',
            kernel_size=17,
            split_ratio=0.25,
            use_ea=True,
        )
    elif args.arch == 'lkmn':
        net_g_opt = dict(
            type='LKMN',
            in_channels=3,
            channels=args.channels,
            out_channels=3,
            upscale=4,
            num_block=args.num_block,
            large_kernel=args.large_kernel,
            split_group=args.split_group,
        )
    elif args.arch in ('spanflex', 'span'):
        return _load_span_model(weights, device, args)
    else:
        raise ValueError(f'Unsupported arch: {args.arch}')

    net_g = build_network(net_g_opt)
    net_g.to(device)
    net_g.eval()

    ckpt = torch.load(weights, map_location=device)
    params = ckpt.get('params_ema', ckpt.get('params', ckpt))
    net_g.load_state_dict(params, strict=True)
    return net_g


def set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--input', required=True, help='Input LR folder')
    parser.add_argument('--output', required=True, help='Output SR folder')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--arch', choices=['plksr_rep', 'lkmn', 'spanflex', 'span'], default='plksr_rep')

    # PLKSR_Rep args
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--n_blocks', type=int, default=9)

    # LKMN args
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--num_block', type=int, default=7)
    parser.add_argument('--large_kernel', type=int, default=21)
    parser.add_argument('--split_group', type=int, default=4)
    parser.add_argument('--block_type', choices=['spab', 'rg', 'rgm'], default='rg')
    parser.add_argument('--span_root', default='/mnt/e/dengjiahao/SPAN')

    parser.add_argument('--max_images', type=int, default=0, help='Limit number of images (0=all)')
    parser.add_argument('--sample', choices=['first', 'random'], default='first')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--prepad',
        type=int,
        default=0,
        help='Reflect-pad LR input by N pixels before inference, then crop SR output back.',
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    set_determinism(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = get_root_logger()
    logger.info(f'Loading {args.arch} weights: {args.weights}')
    net_g = load_model(args.weights, device, args)

    img_paths = sorted(glob(os.path.join(args.input, '*.png')))
    if not img_paths:
        raise FileNotFoundError(f'No PNG files found in {args.input}')

    if args.max_images > 0 and args.max_images < len(img_paths):
        if args.sample == 'random':
            random.seed(args.seed)
            img_paths = random.sample(img_paths, args.max_images)
            img_paths = sorted(img_paths)
        else:
            img_paths = img_paths[: args.max_images]

    with torch.no_grad():
        for path in img_paths:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f'Skip unreadable file: {path}')
                continue
            img = img.astype('float32') / 255.0
            lq = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
            pad = max(0, int(args.prepad))
            if pad > 0:
                lq = F.pad(lq, (pad, pad, pad, pad), mode='reflect')

            with torch.cuda.amp.autocast(enabled=args.fp16):
                out = net_g(lq)
            if pad > 0:
                # x4 SR task in this repo, crop back padded borders in SR space.
                ps = pad * 4
                out = out[:, :, ps:-ps, ps:-ps]

            sr = tensor2img(out, min_max=(0, 1))
            out_name = os.path.basename(path)
            cv2.imwrite(os.path.join(args.output, out_name), sr)

    logger.info(f'Done. Saved to {args.output}')


if __name__ == '__main__':
    main()
