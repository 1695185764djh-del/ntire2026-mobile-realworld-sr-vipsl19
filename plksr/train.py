# flake8: noqa
import os.path as osp

import plksr.archs
import plksr.models
import plksr.data
import plksr.losses
import basicsr.utils.options as options_mod
from basicsr.train import train_pipeline


_orig_expanduser = options_mod.osp.expanduser


def _expanduser_safe(path):
    if isinstance(path, (list, tuple)):
        return [_orig_expanduser(p) for p in path]
    return _orig_expanduser(path)


options_mod.osp.expanduser = _expanduser_safe


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
