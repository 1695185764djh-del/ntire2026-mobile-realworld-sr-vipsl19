import torch
import torch.nn.functional as F
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY


class RepPlainConv2d(nn.Module):
    """
    Train-time 3x3 + 1x1 (+ optional identity) re-parameterizable conv.
    Deploy-time: a single 3x3 conv.
    """

    def __init__(self, c_in, c_out, bias=True, deploy=False, use_identity=False):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.deploy = deploy
        self.use_identity = bool(use_identity and c_in == c_out)

        if deploy:
            self.rep_conv = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=True)
        else:
            self.rbr_3x3 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=bias)
            self.rbr_1x1 = nn.Conv2d(c_in, c_out, 1, stride=1, padding=0, bias=bias)

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel_1x1):
        return F.pad(kernel_1x1, [1, 1, 1, 1])

    def _identity_kernel_bias(self):
        if not self.use_identity:
            return 0, 0
        kernel = torch.zeros(
            self.c_out,
            self.c_in,
            3,
            3,
            device=self.rbr_3x3.weight.device,
            dtype=self.rbr_3x3.weight.dtype,
        )
        for i in range(self.c_out):
            kernel[i, i, 1, 1] = 1.0
        bias = torch.zeros(self.c_out, device=kernel.device, dtype=kernel.dtype)
        return kernel, bias

    def get_equivalent_kernel_bias(self):
        k3 = self.rbr_3x3.weight
        b3 = self.rbr_3x3.bias if self.rbr_3x3.bias is not None else torch.zeros_like(k3[:, 0, 0, 0])
        k1 = self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        b1 = self.rbr_1x1.bias if self.rbr_1x1.bias is not None else torch.zeros_like(b3)
        kid, bid = self._identity_kernel_bias()
        return k3 + k1 + kid, b3 + b1 + bid

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rep_conv = nn.Conv2d(self.c_in, self.c_out, 3, stride=1, padding=1, bias=True)
        self.rep_conv.weight.data.copy_(kernel)
        self.rep_conv.bias.data.copy_(bias)
        del self.rbr_3x3
        del self.rbr_1x1
        self.deploy = True

    def forward(self, x):
        if self.deploy:
            return self.rep_conv(x)
        out = self.rbr_3x3(x) + self.rbr_1x1(x)
        if self.use_identity:
            out = out + x
        return out


class RepPlainBlock(nn.Module):
    """A hardware-friendly residual block: ReparamConv + ReLU + ReparamConv."""

    def __init__(self, channels, deploy=False):
        super().__init__()
        self.conv1 = RepPlainConv2d(channels, channels, deploy=deploy, use_identity=True)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = RepPlainConv2d(channels, channels, deploy=deploy, use_identity=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return out + x


@ARCH_REGISTRY.register()
class RepPlainSR(nn.Module):
    """
    Speed-first plain SR baseline.
    Inference graph is pure conv + ReLU + pixelshuffle after re-parameterization.
    """

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        feature_channels=32,
        num_blocks=4,
        upscale=4,
        deploy=False,
    ):
        super().__init__()
        self.head = RepPlainConv2d(num_in_ch, feature_channels, deploy=deploy, use_identity=False)
        self.body = nn.ModuleList([RepPlainBlock(feature_channels, deploy=deploy) for _ in range(int(num_blocks))])
        self.body_conv = RepPlainConv2d(feature_channels, feature_channels, deploy=deploy, use_identity=False)
        self.upsampler = nn.Sequential(
            nn.Conv2d(feature_channels, num_out_ch * (upscale**2), kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(upscale),
        )

    def forward(self, x):
        feat = self.head(x)
        res = feat
        for block in self.body:
            res = block(res)
        res = self.body_conv(res)
        res = res + feat
        return self.upsampler(res)

    @torch.no_grad()
    def switch_to_deploy(self):
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
