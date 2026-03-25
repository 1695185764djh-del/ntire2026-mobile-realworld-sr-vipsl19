from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY

try:
    import pyiqa
except ModuleNotFoundError:
    pyiqa = None


@LOSS_REGISTRY.register()
class IQACompositeLoss(nn.Module):
    """Composite IQA loss based on pyiqa metrics (e.g., lpips, dists).

    Args:
        metrics (dict): metric name -> weight.
        loss_weight (float): overall scale.
        as_loss (bool): use pyiqa as loss (if supported).
    """

    def __init__(
        self,
        metrics: Dict[str, float],
        loss_weight: float = 1.0,
        as_loss: bool = True,
        higher_is_better: Optional[Dict[str, bool]] = None,
        normalize: Optional[Dict[str, float]] = None,
        clamp_max: Optional[Dict[str, float]] = None,
        fr_resize_to: Optional[int] = None,
        nr_resize_to: Optional[int] = None,
    ):
        super().__init__()
        if pyiqa is None:
            raise ModuleNotFoundError(
                'pyiqa is not installed. Please install it to use IQACompositeLoss.'
            )
        if not metrics or not isinstance(metrics, dict):
            raise ValueError('metrics must be a non-empty dict, e.g. {\"lpips\": 1.0, \"dists\": 1.0}')

        self.loss_weight = float(loss_weight)
        self.metrics = nn.ModuleDict()
        self.weights = {}
        self.metric_modes = {}
        self.higher_is_better = {}
        self.normalize = normalize or {}
        self.clamp_max = clamp_max or {}
        self.fr_resize_to = int(fr_resize_to) if fr_resize_to is not None else None
        self.nr_resize_to = int(nr_resize_to) if nr_resize_to is not None else None
        if self.fr_resize_to is not None and self.fr_resize_to <= 0:
            raise ValueError('fr_resize_to must be a positive int')
        if self.nr_resize_to is not None and self.nr_resize_to <= 0:
            raise ValueError('nr_resize_to must be a positive int')

        # Metrics with larger value indicating better quality.
        default_higher_better = {"clipiqa", "maniqa", "musiq"}
        hb_cfg = higher_is_better or {}

        for name, weight in metrics.items():
            key = name.lower()
            metric = pyiqa.create_metric(key, as_loss=as_loss)
            self.metrics[key] = metric
            self.weights[key] = float(weight)
            self.metric_modes[key] = str(getattr(metric, "metric_mode", "FR")).upper()
            self.higher_is_better[key] = bool(hb_cfg.get(key, key in default_higher_better))
        self._device = None

    def _ensure_device(self, device: torch.device):
        if self._device is None or self._device != device:
            for m in self.metrics.values():
                m.to(device)
            self._device = device

    def _to_minimization_loss(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if name in self.normalize:
            denom = float(self.normalize[name])
            if denom > 0:
                value = value / denom
        if name in self.clamp_max:
            value = torch.clamp(value, max=float(self.clamp_max[name]))
        if self.higher_is_better.get(name, False):
            value = 1.0 - value
        return value

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        self._ensure_device(pred.device)
        total = pred.new_tensor(0.0)
        for name, metric in self.metrics.items():
            mode = self.metric_modes.get(name, "FR")
            if mode == "NR":
                x = pred
                if self.nr_resize_to is not None:
                    x = F.interpolate(x, size=(self.nr_resize_to, self.nr_resize_to), mode='bilinear', align_corners=False)
                value = metric(x)
            else:
                x = pred
                y = target
                if self.fr_resize_to is not None:
                    x = F.interpolate(x, size=(self.fr_resize_to, self.fr_resize_to), mode='bilinear', align_corners=False)
                    y = F.interpolate(y, size=(self.fr_resize_to, self.fr_resize_to), mode='bilinear', align_corners=False)
                value = metric(x, y)
            if isinstance(value, tuple):
                value = value[0]
            value = self._to_minimization_loss(name, value)
            total = total + self.weights[name] * value
        total = total * self.loss_weight
        return total, None
