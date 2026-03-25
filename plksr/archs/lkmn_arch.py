import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


class CA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.pool(x)) * x


class PLKB(nn.Module):
    """Partial large-kernel block used in LKMN."""

    def __init__(self, channels, large_kernel, split_group):
        super().__init__()
        self.channels = channels
        self.split_group = split_group
        self.split_channels = channels // split_group
        self.ca = CA(channels)

        self.dw_kx1 = nn.Conv2d(
            self.split_channels,
            self.split_channels,
            kernel_size=(large_kernel, 1),
            stride=1,
            padding=(large_kernel // 2, 0),
            groups=self.split_channels,
        )
        self.dw_1xk = nn.Conv2d(
            self.split_channels,
            self.split_channels,
            kernel_size=(1, large_kernel),
            stride=1,
            padding=(0, large_kernel // 2),
            groups=self.split_channels,
        )
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.reshape(b, self.split_channels, self.split_group, h, w)
        x = x.permute(0, 2, 1, 3, 4).reshape(b, c, h, w)

        x1, x2 = torch.split(x, (self.split_channels, self.channels - self.split_channels), dim=1)
        x1 = self.ca(x1)
        x1 = self.dw_kx1(self.dw_1xk(x1))
        out = torch.cat((x1, x2), dim=1)
        return self.act(self.conv1(out))


class HFAB(nn.Module):
    def __init__(self, channels, large_kernel, split_group):
        super().__init__()
        self.plkb = PLKB(channels, large_kernel, split_group)
        self.dw3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.conv1 = nn.Conv2d(channels * 2, channels, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.dw3(x)
        x2 = self.plkb(x)
        return self.act(self.conv1(torch.cat((x1, x2), dim=1)))


class HFDB(nn.Module):
    def __init__(self, channels, large_kernel, split_group):
        super().__init__()
        self.c1_d = nn.Conv2d(channels, channels // 2, 1)
        self.c1_r = HFAB(channels, large_kernel, split_group)
        self.c2_d = nn.Conv2d(channels, channels // 2, 1)
        self.c2_r = HFAB(channels, large_kernel, split_group)
        self.c3_d = nn.Conv2d(channels, channels // 2, 1)
        self.c3_r = HFAB(channels, large_kernel, split_group)
        self.c4 = nn.Conv2d(channels, channels // 2, 1)
        self.c5 = nn.Conv2d(channels * 2, channels, 1)
        self.act = nn.GELU()

    def forward(self, x):
        d1 = self.act(self.c1_d(x))
        r1 = self.c1_r(x)
        d2 = self.act(self.c2_d(r1))
        r2 = self.c2_r(r1)
        d3 = self.act(self.c3_d(r2))
        r3 = self.c3_r(r2)
        r4 = self.act(self.c4(r3))
        out = torch.cat([d1, d2, d3, r4], dim=1)
        return self.act(self.c5(out))


class Scaler(nn.Module):
    def __init__(self, channels, init_value=1e-5, requires_grad=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(1, channels, 1, 1), requires_grad=requires_grad)

    def forward(self, x):
        return x * self.scale


class CGFN(nn.Module):
    def __init__(self, channels, large_kernel, split_group):
        super().__init__()
        self.plkb = PLKB(channels, large_kernel, split_group)
        self.dw3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0)
        self.scaler1 = Scaler(channels)
        self.scaler2 = Scaler(channels)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.plkb(x)
        x1_s = self.scaler1(x - x1)
        x2 = self.dw3(x)
        x2_s = self.scaler2(x - x2)
        x1 = x1 * x2_s
        x2 = x2 * x1_s
        return self.act(self.conv1(torch.cat((x1, x2), dim=1)))


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class RFMG(nn.Module):
    def __init__(self, channels, large_kernel, split_group):
        super().__init__()
        self.hfdb = HFDB(channels, large_kernel, split_group)
        self.cgfn = CGFN(channels, large_kernel, split_group)
        self.norm1 = LayerNorm(channels, data_format="channels_first")
        self.norm2 = LayerNorm(channels, data_format="channels_first")

    def forward(self, x):
        x = self.hfdb(self.norm1(x)) + x
        x = self.cgfn(self.norm2(x)) + x
        return x


@ARCH_REGISTRY.register()
class LKMN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        channels=36,
        out_channels=3,
        upscale=4,
        num_block=8,
        large_kernel=31,
        split_group=4,
    ):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)
        self.layers = nn.Sequential(*[RFMG(channels, large_kernel, split_group) for _ in range(num_block)])
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.upsampler = nn.Sequential(
            nn.Conv2d(channels, (upscale**2) * out_channels, 3, 1, 1),
            nn.PixelShuffle(upscale),
        )
        self.act = nn.GELU()

    def forward(self, x):
        out_fea = self.conv_first(x)
        out = self.layers(out_fea)
        out = self.act(self.conv(out))
        return self.upsampler(out + out_fea)
