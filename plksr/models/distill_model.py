import torch
import torch.nn.functional as F
from collections import OrderedDict
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class DistillSRModel(SRModel):
    """支持知识蒸馏的超分模型类"""

    @staticmethod
    def _resolve_module(root_module, module_path):
        module = root_module
        for token in module_path.split('.'):
            if token.lstrip('-').isdigit():
                module = module[int(token)]
            else:
                module = getattr(module, token)
        return module

    @staticmethod
    def _build_feature_hook(cache, name):
        def _hook(_module, _inputs, output):
            cache[name] = output
        return _hook

    def init_training_settings(self):
        # 1. 初始化标准的 Student 训练设置 (继承自 SRModel)
        super().init_training_settings()
        
        train_opt = self.opt['train']

        # 2. 初始化蒸馏损失 (Distillation Loss)
        if train_opt.get('distill_opt'):
            self.cri_distill = build_loss(train_opt['distill_opt']).to(self.device)
        else:
            self.cri_distill = None
        self.distill_weight = float(train_opt.get('distill_weight', 1.0))
            
        # 3. 加载 Teacher 网络
        self.net_g_teacher = build_network(self.opt['network_g_teacher'])
        self.net_g_teacher = self.model_to_device(self.net_g_teacher)
        
        # 4. 加载 Teacher 权重
        load_path = self.opt['path'].get('pretrain_network_g_teacher')
        if load_path is not None:
            logger = get_root_logger()
            logger.info(f'Loading teacher model from {load_path}')
            # 简单的权重加载逻辑，假设是pth文件
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            if 'params-ema' in checkpoint:
                keyname = 'params-ema'
            elif 'params' in checkpoint:
                keyname = 'params'
            else:
                keyname = None
            
            param_dict = checkpoint[keyname] if keyname else checkpoint
            self.net_g_teacher.load_state_dict(param_dict, strict=True)
        
        # 5. 冻结 Teacher，不让它更新
        self.net_g_teacher.eval()
        for param in self.net_g_teacher.parameters():
            param.requires_grad = False

        # 6. [可选] 特征蒸馏 (借鉴 ESRModel 的做法)
        self.cri_feat_distill = None
        self.feat_distill_layers = []
        self.feat_distill_weights = []
        self._feat_s = {}
        self._feat_t = {}
        self._feat_hooks = []
        if train_opt.get('feat_distill_opt'):
            self.cri_feat_distill = build_loss(train_opt['feat_distill_opt']).to(self.device)
            layers = train_opt.get('feat_distill_layers', [])
            if isinstance(layers, str):
                layers = [layers]
            self.feat_distill_layers = list(layers)

            weights = train_opt.get('feat_distill_weight', 1.0)
            if isinstance(weights, (int, float)):
                self.feat_distill_weights = [float(weights)] * len(self.feat_distill_layers)
            else:
                self.feat_distill_weights = [float(v) for v in weights]
                if len(self.feat_distill_weights) != len(self.feat_distill_layers):
                    raise ValueError('feat_distill_weight length must match feat_distill_layers length.')

            for layer_name in self.feat_distill_layers:
                s_mod = self._resolve_module(self.net_g, layer_name)
                t_mod = self._resolve_module(self.net_g_teacher, layer_name)
                self._feat_hooks.append(s_mod.register_forward_hook(self._build_feature_hook(self._feat_s, layer_name)))
                self._feat_hooks.append(t_mod.register_forward_hook(self._build_feature_hook(self._feat_t, layer_name)))
            logger.info(f'Enable feature distillation on layers: {self.feat_distill_layers}')

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        if self.cri_feat_distill is not None:
            self._feat_s.clear()
            self._feat_t.clear()
        
        # --- 前向传播 ---
        # 1. Student 跑一遍
        self.output = self.net_g(self.lq)
        
        # 2. Teacher 跑一遍 (不计算梯度)
        with torch.no_grad():
            self.output_teacher = self.net_g_teacher(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        # --- 计算 Loss ---
        # 1. Pixel Loss (Student vs GT)
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # 2. Distillation Loss (Student vs Teacher)
        if self.cri_distill:
            l_distill = self.cri_distill(self.output, self.output_teacher)
            l_distill = l_distill * self.distill_weight
            l_total += l_distill
            loss_dict['l_distill'] = l_distill

        # 2.5 Feature Distillation Loss
        if self.cri_feat_distill is not None:
            l_feat = 0
            feat_count = 0
            for layer_name, weight in zip(self.feat_distill_layers, self.feat_distill_weights):
                feat_s = self._feat_s.get(layer_name)
                feat_t = self._feat_t.get(layer_name)
                if feat_s is None or feat_t is None:
                    continue
                if feat_s.shape[-2:] != feat_t.shape[-2:]:
                    feat_t = F.interpolate(feat_t, size=feat_s.shape[-2:], mode='bilinear', align_corners=False)
                if feat_s.shape[1] != feat_t.shape[1]:
                    ch = min(feat_s.shape[1], feat_t.shape[1])
                    feat_s = feat_s[:, :ch]
                    feat_t = feat_t[:, :ch]
                l_feat = l_feat + self.cri_feat_distill(feat_s, feat_t) * weight
                feat_count += 1
            if feat_count > 0:
                l_total += l_feat
                loss_dict['l_feat'] = l_feat

        # 3. Perceptual Loss (如果有配置的话)
        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
            if l_g_percep is not None:
                l_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_total += l_g_style
                loss_dict['l_g_style'] = l_g_style

        # --- 反向传播 ---
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # EMA 更新
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
