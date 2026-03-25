import torch
import torch.nn.functional as F
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY


def _as_pair(value):
    if isinstance(value, int):
        return (value, value)
    return value


def _conv_layer(in_channels, out_channels, kernel_size, bias=True):
    k = _as_pair(kernel_size)
    padding = ((k[0] - 1) // 2, (k[1] - 1) // 2)
    return nn.Conv2d(in_channels, out_channels, k, padding=padding, bias=bias)


def _pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3):
    conv = _conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size)
    return nn.Sequential(conv, nn.PixelShuffle(upscale_factor))


class Conv3XC(nn.Module):
    """Minimal Conv3XC block matching team23 state_dict key layout."""

    def __init__(self, c_in, c_out, s=1, bias=True, relu=False):
        super().__init__()
        self.has_relu = relu
        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        )

    def forward(self, x):
        out = self.eval_conv(x)
        if self.has_relu:
            out = torch.nn.functional.leaky_relu(out, negative_slope=0.05)
        return out


class RepConv2d(nn.Module):
    """Train-time multi-branch conv; deploy-time single 3x3 conv."""

    def __init__(self, c_in, c_out, s=1, bias=True, deploy=False, use_identity=False):
        super().__init__()
        self.deploy = deploy
        self.c_in = c_in
        self.c_out = c_out
        self.s = s
        self.use_identity = use_identity and c_in == c_out and s == 1

        if deploy:
            self.eval_conv = nn.Conv2d(c_in, c_out, 3, stride=s, padding=1, bias=True)
        else:
            self.rbr_3x3 = nn.Conv2d(c_in, c_out, 3, stride=s, padding=1, bias=bias)
            self.rbr_1x1 = nn.Conv2d(c_in, c_out, 1, stride=s, padding=0, bias=bias)
            self.rbr_identity = nn.Identity() if self.use_identity else None

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel_1x1):
        return F.pad(kernel_1x1, [1, 1, 1, 1])

    def _identity_kernel_bias(self):
        if not self.use_identity:
            return 0, 0
        kernel = torch.zeros(
            self.c_out, self.c_in, 3, 3, device=self.rbr_3x3.weight.device, dtype=self.rbr_3x3.weight.dtype
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
        self.eval_conv = nn.Conv2d(self.c_in, self.c_out, 3, stride=self.s, padding=1, bias=True)
        self.eval_conv.weight.data.copy_(kernel)
        self.eval_conv.bias.data.copy_(bias)
        del self.rbr_3x3
        del self.rbr_1x1
        if self.rbr_identity is not None:
            del self.rbr_identity
        self.deploy = True

    def forward(self, x):
        if self.deploy:
            return self.eval_conv(x)
        out = self.rbr_3x3(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out = out + x
        return out


def _build_conv3x(c_in, c_out, s=1, bias=True, conv_mode="plain", deploy=False):
    if conv_mode == "plain":
        return Conv3XC(c_in, c_out, s=s, bias=bias)
    if conv_mode == "rep":
        # Disable extra identity branch by default to avoid unstable amplification in SPAB stack.
        return RepConv2d(c_in, c_out, s=s, bias=bias, deploy=deploy, use_identity=False)
    raise ValueError(f"Unknown conv mode: {conv_mode}")


def _normalize_block_type(name):
    key = str(name).lower()
    if key in ("spab", "default"):
        return "spab"
    if key in ("rg", "repgate", "repgated"):
        return "rg"
    if key in ("rgm", "msrg", "multiscale-repgate"):
        return "rgm"
    if key in ("lite", "litespab"):
        return "lite"
    if key in ("auto",):
        return "auto"
    raise ValueError(f"Unknown block_type: {name}")


class SPAB(nn.Module):
    """SPAN-like lightweight attention block used by DSCF."""

    def __init__(
        self,
        in_channels,
        mid_channels=None,
        out_channels=None,
        bias=False,
        conv_mode="plain",
        gate_type="sigmoid",
        deploy=False,
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.gate_type = gate_type
        self.c1_r = _build_conv3x(in_channels, mid_channels, s=1, bias=bias, conv_mode=conv_mode, deploy=deploy)
        self.c2_r = _build_conv3x(mid_channels, mid_channels, s=1, bias=bias, conv_mode=conv_mode, deploy=deploy)
        self.c3_r = _build_conv3x(mid_channels, out_channels, s=1, bias=bias, conv_mode=conv_mode, deploy=deploy)
        self.act1 = nn.SiLU(inplace=True)

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)
        if self.gate_type == "sigmoid":
            sim_att = torch.sigmoid(out3) - 0.5
        elif self.gate_type == "hardsigmoid":
            sim_att = F.hardsigmoid(out3) - 0.5
        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")
        out = (out3 + x) * sim_att
        return out, out1, out2, out3


class RepGate(nn.Module):
    """Depthwise-pointwise gate used by SPAB_RG."""

    def __init__(self, channels, bias=True, dw_kernel_size=3):
        super().__init__()
        pad = dw_kernel_size // 2
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=dw_kernel_size,
            padding=pad,
            groups=channels,
            bias=bias,
        )
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=bias)
        if self.pw.bias is not None:
            nn.init.constant_(self.pw.bias, 1.0)

    def forward(self, x):
        return torch.sigmoid(self.dw(x) + self.pw(x))


class MultiScaleRepGate(nn.Module):
    """Two-branch depthwise gate used by SPAB_RGM."""

    def __init__(self, channels, bias=True, dw_kernel_size=5):
        super().__init__()
        pad = dw_kernel_size // 2
        self.dw3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=bias)
        self.dwk = nn.Conv2d(
            channels,
            channels,
            kernel_size=dw_kernel_size,
            padding=pad,
            groups=channels,
            bias=bias,
        )
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=bias)
        if self.pw.bias is not None:
            nn.init.constant_(self.pw.bias, 1.0)

    def forward(self, x):
        return torch.sigmoid(self.dw3(x) + self.dwk(x) + self.pw(x))


class SPAB_RG(nn.Module):
    """SPAB body with RepGate gating."""

    def __init__(
        self,
        in_channels,
        mid_channels=None,
        out_channels=None,
        bias=False,
        conv_mode="plain",
        deploy=False,
        gate_dw_kernel_size=3,
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.c1_r = _build_conv3x(in_channels, mid_channels, s=1, bias=bias, conv_mode=conv_mode, deploy=deploy)
        self.c2_r = _build_conv3x(mid_channels, mid_channels, s=1, bias=bias, conv_mode=conv_mode, deploy=deploy)
        self.c3_r = _build_conv3x(mid_channels, out_channels, s=1, bias=bias, conv_mode=conv_mode, deploy=deploy)
        self.act1 = nn.SiLU(inplace=True)
        self.gate = RepGate(out_channels, bias=True, dw_kernel_size=gate_dw_kernel_size)

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)
        gate = self.gate(out3 + x)
        out = (out3 + x) * gate
        return out, out1, out2, out3


class SPAB_RGM(nn.Module):
    """SPAB body with multi-scale RepGate gating."""

    def __init__(
        self,
        in_channels,
        mid_channels=None,
        out_channels=None,
        bias=False,
        conv_mode="plain",
        deploy=False,
        rgm_dw_kernel_size=5,
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.c1_r = _build_conv3x(in_channels, mid_channels, s=1, bias=bias, conv_mode=conv_mode, deploy=deploy)
        self.c2_r = _build_conv3x(mid_channels, mid_channels, s=1, bias=bias, conv_mode=conv_mode, deploy=deploy)
        self.c3_r = _build_conv3x(mid_channels, out_channels, s=1, bias=bias, conv_mode=conv_mode, deploy=deploy)
        self.act1 = nn.SiLU(inplace=True)
        self.gate = MultiScaleRepGate(out_channels, bias=True, dw_kernel_size=rgm_dw_kernel_size)

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)
        gate = self.gate(out3 + x)
        out = (out3 + x) * gate
        return out, out1, out2, out3


class LiteSPAB(nn.Module):
    """
    Lightweight tail block:
    depthwise + pointwise stack with the same output interface as SPAB.
    """

    def __init__(self, in_channels, bias=False, gate_type="sigmoid", dw_kernel_size=5):
        super().__init__()
        pad = dw_kernel_size // 2
        self.gate_type = gate_type

        self.dw1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=dw_kernel_size,
            stride=1,
            padding=pad,
            groups=in_channels,
            bias=bias,
        )
        self.pw1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.dw2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=bias,
        )
        self.pw2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        out1 = self.pw1(self.act(self.dw1(x)))
        out2 = self.pw2(self.act(self.dw2(out1)))
        out3 = out2

        if self.gate_type == "sigmoid":
            sim_att = torch.sigmoid(out3) - 0.5
        elif self.gate_type == "hardsigmoid":
            sim_att = F.hardsigmoid(out3) - 0.5
        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")
        out = (out3 + x) * sim_att
        return out, out1, out2, out3


@ARCH_REGISTRY.register()
class DSCFSR(nn.Module):
    """
    Configurable DSCF architecture compatible with team23 checkpoint naming.

    The original model is feature_channels=26, num_blocks=6.
    """

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        feature_channels=26,
        num_blocks=6,
        upscale=4,
        bias=True,
        conv_mode="plain",
        gate_type="sigmoid",
        deploy=False,
        block_type="spab",
        tail_block_type="auto",
        tail_replace_blocks=0,
        tail_lite_blocks=0,
        gate_dw_kernel_size=3,
        rgm_dw_kernel_size=5,
        lite_dw_kernel_size=5,
        img_range=255.0,
        rgb_mean=(0.4488, 0.4371, 0.4040),
    ):
        super().__init__()
        if num_blocks < 3:
            raise ValueError("num_blocks must be >= 3 for stable DSCF-style aggregation.")
        if tail_lite_blocks < 0 or tail_lite_blocks > num_blocks:
            raise ValueError("tail_lite_blocks must be in [0, num_blocks].")
        if tail_replace_blocks < 0 or tail_replace_blocks > num_blocks:
            raise ValueError("tail_replace_blocks must be in [0, num_blocks].")

        self.img_range = float(img_range)
        self.num_blocks = int(num_blocks)
        self.tail_lite_blocks = int(tail_lite_blocks)
        self.tail_replace_blocks = int(tail_replace_blocks)
        self.block_type = _normalize_block_type(block_type)
        self.tail_block_type = _normalize_block_type(tail_block_type)
        if self.tail_block_type == "auto":
            self.tail_block_type = self.block_type
        # Keep this buffer out of checkpoint matching so original team23 weights load strictly.
        self.register_buffer("mean", torch.tensor(rgb_mean).view(1, 3, 1, 1), persistent=False)

        self.conv_1 = _build_conv3x(num_in_ch, feature_channels, s=1, bias=bias, conv_mode=conv_mode, deploy=deploy)

        def make_block(block_name):
            if block_name == "spab":
                return SPAB(
                    feature_channels,
                    bias=bias,
                    conv_mode=conv_mode,
                    gate_type=gate_type,
                    deploy=deploy,
                )
            if block_name == "rg":
                return SPAB_RG(
                    feature_channels,
                    bias=bias,
                    conv_mode=conv_mode,
                    deploy=deploy,
                    gate_dw_kernel_size=gate_dw_kernel_size,
                )
            if block_name == "rgm":
                return SPAB_RGM(
                    feature_channels,
                    bias=bias,
                    conv_mode=conv_mode,
                    deploy=deploy,
                    rgm_dw_kernel_size=rgm_dw_kernel_size,
                )
            if block_name == "lite":
                return LiteSPAB(
                    feature_channels,
                    bias=bias,
                    gate_type=gate_type,
                    dw_kernel_size=lite_dw_kernel_size,
                )
            raise ValueError(f"Unsupported normalized block type: {block_name}")

        for idx in range(self.num_blocks):
            is_lite_tail = idx >= (self.num_blocks - self.tail_lite_blocks)
            if is_lite_tail:
                block = make_block("lite")
            else:
                use_tail_replacement = idx >= (self.num_blocks - self.tail_replace_blocks)
                block_name = self.tail_block_type if use_tail_replacement else self.block_type
                block = make_block(block_name)
            setattr(self, f"block_{idx + 1}", block)

        self.conv_cat = _conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        self.conv_2 = _build_conv3x(
            feature_channels, feature_channels, s=1, bias=bias, conv_mode=conv_mode, deploy=deploy
        )
        self.upsampler = _pixelshuffle_block(feature_channels, num_out_ch, upscale_factor=upscale)

    def _block(self, idx):
        return getattr(self, f"block_{idx + 1}")

    def forward(self, x):
        mean = self.mean.type_as(x)
        x = (x - mean) * self.img_range

        out_feature = self.conv_1(x)
        out = out_feature
        out_b1 = None
        out_anchor = None
        for idx in range(self.num_blocks):
            out, out1, _, _ = self._block(idx)(out)
            if idx == 0:
                out_b1 = out
            if idx == self.num_blocks - 1:
                out_anchor = out1

        out_tail = self.conv_2(out)
        out = self.conv_cat(torch.cat([out_feature, out_tail, out_b1, out_anchor], dim=1))
        return self.upsampler(out)

    @torch.no_grad()
    def switch_to_deploy(self):
        for m in self.modules():
            if m is self:
                continue
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()
