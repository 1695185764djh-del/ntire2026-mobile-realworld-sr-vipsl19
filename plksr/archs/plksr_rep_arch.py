import torch
from torch import nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from .plksr_arch import PLKBlock, PLKSR, DCCM, EA 

# =========================================================================
# 1. 定义 RepConv (保持不变)
# =========================================================================
class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(RepConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.bias = bias
        
        assert kernel_size == 3, "RepConv currently optimized for 3x3"
        
        self.rbr_dense = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups, bias=bias)
        self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, groups=groups, bias=bias)
        self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
        self.deploy = False

    def forward(self, inputs):
        if self.deploy:
            return self.rbr_reparam(inputs)
        out = self.rbr_dense(inputs) + self.rbr_1x1(inputs)
        if self.rbr_identity is not None:
            out += self.rbr_identity(inputs)
        return out

    def switch_to_deploy(self):
        if self.deploy: return
        kernel_3x3, bias_3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        if hasattr(self, 'rbr_identity') and self.rbr_identity is not None:
            kernel_id, bias_id = self._fuse_bn_tensor(self.rbr_identity)
        else:
            kernel_id, bias_id = 0, 0
            
        pad_1x1 = F.pad(kernel_1x1, [1, 1, 1, 1])
        final_kernel = kernel_3x3 + pad_1x1 + kernel_id
        final_bias = bias_3x3 + bias_1x1 + bias_id

        self.rbr_reparam = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, groups=self.groups, bias=True)
        self.rbr_reparam.weight.data = final_kernel
        self.rbr_reparam.bias.data = final_bias

        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'): self.__delattr__('rbr_identity')
        self.deploy = True

    def _fuse_bn_tensor(self, branch):
        if branch is None: return 0, 0
        if isinstance(branch, nn.Conv2d): return branch.weight.data, branch.bias.data
        elif isinstance(branch, nn.BatchNorm2d):
            input_dim = self.in_channels
            kernel_value = torch.zeros((self.in_channels, input_dim, 3, 3), dtype=branch.weight.dtype, device=branch.weight.device)
            for i in range(self.in_channels): kernel_value[i, i, 1, 1] = 1
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel_value * t, beta - running_mean * gamma / std
        return 0, 0

# =========================================================================
# 2. 定义 RepDCCM (保持不变)
# =========================================================================
class RepDCCM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = RepConv(dim, dim * 2, 3) 
        self.act = nn.GELU()
        self.conv2 = RepConv(dim * 2, dim, 3) 

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))

# =========================================================================
# 3. 定义 PLKSR_Rep (关键修改：外科手术式替换)
# =========================================================================
@ARCH_REGISTRY.register()
class PLKSR_Rep(PLKSR):
    def __init__(self, dim=64, n_blocks=28, upscaling_factor=4, ccm_type='DCCM', kernel_size=17, split_ratio=0.25, lk_type='PLK', use_ea=True):
        # 1. 调用父类初始化，让它把该建的层都建好（包括 Head, Body, Tail）
        super().__init__(
            dim=dim, 
            n_blocks=n_blocks, 
            upscaling_factor=upscaling_factor, 
            ccm_type=ccm_type, 
            kernel_size=kernel_size, 
            split_ratio=split_ratio, 
            lk_type=lk_type,
            use_ea=use_ea
        )
        
        # 2. 遍历现有的 self.feats，只替换 PLKBlock，保留其他层 (如 Head Conv)
        new_layers = []
        for module in self.feats:
            if isinstance(module, PLKBlock):
                # 如果是 Block，就创建一个新的 Rep 版 Block 替换它
                # 第一步：用偷天换日法初始化
                rep_block = PLKBlock(
                    dim=dim, 
                    ccm_type='DCCM',             # 骗过初始化检查
                    max_kernel_size=kernel_size, # 正确的参数名
                    split_ratio=split_ratio, 
                    lk_type=lk_type,       
                    use_ea=use_ea
                )
                # 第二步：植入 RepDCCM
                del rep_block.channe_mixer
                rep_block.channe_mixer = RepDCCM(dim)
                
                new_layers.append(rep_block)
            else:
                # [关键] 如果是 Head 卷积或其他层，原样保留！
                new_layers.append(module)

        # 3. 用重组后的列表覆盖 self.feats
        self.feats = nn.Sequential(*new_layers)