import torch
from torch.nn import functional as F
from basicsr.models import lr_scheduler as lr_scheduler
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel
from basicsr.archs import build_network
from basicsr.metrics import calculate_metric
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img
from torch.optim.lr_scheduler import _LRScheduler
import math
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import os.path as osp

import numpy as np


@MODEL_REGISTRY.register()
class ESRModel(BaseModel):
    def __init__(self, opt):
        super(ESRModel, self).__init__(opt)

        # -------------------------------------------------------
        # 1. Define Student Network (net_g)
        # -------------------------------------------------------
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # Load Student pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            use_ema = self.opt.get('use_ema', False)
            if use_ema:
                param_key = 'params_ema'
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # -------------------------------------------------------
        # 2. [新增] Define Teacher Network (net_teacher)
        # -------------------------------------------------------
        if opt.get('network_g_teacher', None):
            self.net_teacher = build_network(opt['network_g_teacher'])
            self.net_teacher = self.model_to_device(self.net_teacher)
            print('Loading Teacher model...')
            self.print_network(self.net_teacher)
            
            # Load Teacher weights
            load_path_teacher = self.opt['path'].get('pretrain_network_g_teacher', None)
            if load_path_teacher is not None:
                # 使用 BasicSR 的 load_network 加载权重
                self.load_network(self.net_teacher, load_path_teacher, True, 'params')
            else:
                raise FileNotFoundError('Error: pretrain_network_g_teacher path is missing in config!')

            # Freeze Teacher (Set to eval mode and disable gradients)
            self.net_teacher.eval()
            for k, v in self.net_teacher.named_parameters():
                v.requires_grad = False
        # -------------------------------------------------------

        if self.is_train:
            self.init_training_settings()

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
        self.net_g.train()
        self.use_amp = self.opt.get('use_amp', False)
        if self.use_amp:
            logger = get_root_logger()
            logger.info('Use mixed precision training.')
            self.scaler = torch.cuda.amp.GradScaler()
        
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        if train_opt.get('wave_opt'):
            self.cri_wave = build_loss(train_opt['wave_opt']).to(self.device)
        else:
            self.cri_wave = None
        
        if train_opt.get('mesa_opt'):
            start_ratio = train_opt['mesa_opt'].pop('start_ratio', 0.33)
            self.mesa_start_iter = int(start_ratio * train_opt['total_iter'])
            self.cri_mesa = build_loss(train_opt['mesa_opt']).to(self.device)
        else:
            self.cri_mesa = None

        if train_opt.get('distill_opt'):
            self.cri_distill = build_loss(train_opt['distill_opt']).to(self.device)
        else:
            self.cri_distill = None
        self.distill_weight = train_opt.get('distill_weight', 1.0)

        # Optional feature distillation: compare selected internal feature maps.
        self.cri_feat_distill = None
        self.feat_distill_layers = []
        self.feat_distill_weights = []
        self._feat_s = {}
        self._feat_t = {}
        self._feat_hooks = []
        if train_opt.get('feat_distill_opt'):
            if hasattr(self, 'net_teacher'):
                self.cri_feat_distill = build_loss(train_opt['feat_distill_opt']).to(self.device)
                layers = train_opt.get('feat_distill_layers', ['feats.10'])
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
                    t_mod = self._resolve_module(self.net_teacher, layer_name)
                    self._feat_hooks.append(s_mod.register_forward_hook(self._build_feature_hook(self._feat_s, layer_name)))
                    self._feat_hooks.append(t_mod.register_forward_hook(self._build_feature_hook(self._feat_t, layer_name)))

                logger = get_root_logger()
                logger.info(f'Use feature distillation on layers: {self.feat_distill_layers}')
            else:
                logger = get_root_logger()
                logger.warning('feat_distill_opt is set but no teacher network found. Feature distillation is disabled.')

        # [Note] If using distillation only, you might technically not need pixel loss, 
        # but usually we combine them.
        if self.cri_pix is None and self.cri_perceptual is None and self.cri_fft is None and self.cri_wave is None:
            # raise ValueError('Pixel, perceptual and frequency losses are None.')
            pass 

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        
    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingLR(optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')
    
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        # ------------------------------------------------------
        # CASE 1: Normal Training (No Mixed Precision)
        # ------------------------------------------------------
        if not self.use_amp:
            self.optimizer_g.zero_grad()
            if self.cri_feat_distill is not None:
                self._feat_s.clear()
                self._feat_t.clear()
            
            # 1. Student Forward
            self.output = self.net_g(self.lq)

            # 2. [新增] Teacher Forward (No Grad)
            if hasattr(self, 'net_teacher'):
                with torch.no_grad():
                    self.output_teacher = self.net_teacher(self.lq)

            l_total = 0
            loss_dict = OrderedDict()

            # (A) Pixel Loss
            if self.cri_pix:
                l_pix = self.cri_pix(self.output, self.gt)
                l_total += l_pix
                loss_dict['l_pix'] = l_pix

            # (B) Frequency / Wavelet / Perceptual / Mesa Losses
            if self.cri_fft:
                l_fft = self.cri_fft(self.output, self.gt)
                l_total += l_fft
                loss_dict['l_freq'] = l_fft
            if self.cri_wave:
                l_wave = self.cri_wave(self.output, self.gt)
                l_total += l_wave
                loss_dict['l_wave'] = l_wave
            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(self.output, self.gt)
                if l_percep is not None:
                    l_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_total += l_style
                    loss_dict['l_style'] = l_style
            if self.cri_mesa:
                if current_iter >= self.mesa_start_iter:
                    with torch.no_grad():
                        output_emas = self.net_g_ema(self.lq)
                    l_mesa = self.cri_mesa(self.output, output_emas)
                    l_total += l_mesa
                    loss_dict['l_mesa'] = l_mesa
                else:
                    l_mesa = torch.zeros(1, device=self.device)
                    loss_dict['l_mesa'] = l_mesa

            # (C) [新增] Distillation Loss
            if hasattr(self, 'net_teacher'):
                if self.cri_distill:
                    l_distill = self.cri_distill(self.output, self.output_teacher)
                elif self.cri_pix:
                    l_distill = self.cri_pix(self.output, self.output_teacher) * self.distill_weight
                else:
                    l_distill = None
                if l_distill is not None:
                    l_total += l_distill
                    loss_dict['l_distill'] = l_distill

            # (D) Feature Distillation Loss
            if hasattr(self, 'net_teacher') and self.cri_feat_distill is not None:
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

            l_total.backward()
            self.optimizer_g.step()
        
        # ------------------------------------------------------
        # CASE 2: Mixed Precision Training (AMP)
        # ------------------------------------------------------
        else:
            with torch.cuda.amp.autocast():
                self.optimizer_g.zero_grad()
                if self.cri_feat_distill is not None:
                    self._feat_s.clear()
                    self._feat_t.clear()
                
                # 1. Student Forward
                self.output = self.net_g(self.lq)
                
                # 2. [新增] Teacher Forward
                if hasattr(self, 'net_teacher'):
                    with torch.no_grad():
                        self.output_teacher = self.net_teacher(self.lq)

                l_total = 0
                loss_dict = OrderedDict()

                # (A) Pixel Loss
                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                # (B) Other Losses
                if self.cri_fft:
                    l_fft = self.cri_fft(self.output, self.gt)
                    l_total += l_fft
                    loss_dict['l_freq'] = l_fft
                if self.cri_wave:
                    l_wave = self.cri_wave(self.output, self.gt)
                    l_total += l_wave
                    loss_dict['l_wave'] = l_wave
                if self.cri_perceptual:
                    l_percep, l_style = self.cri_perceptual(self.output, self.gt)
                    if l_percep is not None:
                        l_total += l_percep
                        loss_dict['l_percep'] = l_percep
                    if l_style is not None:
                        l_total += l_style
                        loss_dict['l_style'] = l_style
                if self.cri_mesa:
                    if current_iter >= self.mesa_start_iter:
                        with torch.no_grad():
                            output_emas = self.net_g_ema(self.lq)
                        l_mesa = self.cri_mesa(self.output, output_emas)
                        l_total += l_mesa
                        loss_dict['l_mesa'] = l_mesa
                    else:
                        l_mesa = torch.zeros(1, device=self.device)
                        loss_dict['l_mesa'] = l_mesa
                
                # (C) [新增] Distillation Loss
                if hasattr(self, 'net_teacher'):
                    if self.cri_distill:
                        l_distill = self.cri_distill(self.output, self.output_teacher)
                    elif self.cri_pix:
                        l_distill = self.cri_pix(self.output, self.output_teacher) * self.distill_weight
                    else:
                        l_distill = None
                    if l_distill is not None:
                        l_total += l_distill
                        loss_dict['l_distill'] = l_distill

                # (D) Feature Distillation Loss
                if hasattr(self, 'net_teacher') and self.cri_feat_distill is not None:
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
                
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
            
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_opt = dataloader.dataset.opt
        dataset_name = dataset_opt['name']
        # Allow per-dataset override for saving images.
        save_img = dataset_opt.get('save_img', save_img)
        has_gt = 'dataroot_gt' in dataset_opt and dataset_opt.get('dataroot_gt') is not None
        with_metrics = self.opt['val'].get('metrics') is not None and has_gt
        if self.opt['val'].get('metrics') is not None and not has_gt:
            logger = get_root_logger()
            logger.info(f'Validation {dataset_name}: no GT, skip metrics.')
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
