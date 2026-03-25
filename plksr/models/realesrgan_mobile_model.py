from collections import OrderedDict

import torch
from basicsr.losses.loss_util import get_refined_artifact_map
from basicsr.models.realesrgan_model import RealESRGANModel as _RealESRGANModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class RealESRGANMobileModel(_RealESRGANModel):
    """RealESRGANModel with no-GT validation support per dataset.

    If a validation dataset has no GT (e.g., mobile_val), metrics are skipped and
    images can still be saved for submission.
    """

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_opt = dataloader.dataset.opt
        dataset_name = dataset_opt.get('name', 'val')
        save_img = dataset_opt.get('save_img', save_img)

        has_gt = 'dataroot_gt' in dataset_opt and dataset_opt.get('dataroot_gt') is not None
        if self.opt['val'].get('metrics') is not None and not has_gt:
            logger = get_root_logger()
            logger.info(f'Validation {dataset_name}: no GT, skip metrics.')
            orig_metrics = self.opt['val'].get('metrics')
            self.opt['val']['metrics'] = None
            super().nondist_validation(dataloader, current_iter, tb_logger, save_img)
            self.opt['val']['metrics'] = orig_metrics
            return

        super().nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def optimize_parameters(self, current_iter):
        # Stage-1 strict warm-up: train only G before D start.
        if current_iter > self.net_d_init_iters:
            super().optimize_parameters(current_iter)
            return

        l1_gt = self.gt_usm if self.opt['l1_gt_usm'] else self.gt
        percep_gt = self.gt_usm if self.opt['percep_gt_usm'] else self.gt

        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        if self.cri_ldl:
            self.output_ema = self.net_g_ema(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_g_pix = self.cri_pix(self.output, l1_gt)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix

        if self.cri_ldl:
            pixel_weight = get_refined_artifact_map(self.gt, self.output, self.output_ema, 7)
            l_g_ldl = self.cri_ldl(torch.mul(pixel_weight, self.output), torch.mul(pixel_weight, self.gt))
            l_g_total += l_g_ldl
            loss_dict['l_g_ldl'] = l_g_ldl

        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
            if l_g_percep is not None:
                l_g_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_g_total += l_g_style
                loss_dict['l_g_style'] = l_g_style

        l_g_total.backward()
        self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
