from copy import deepcopy
from os import path as osp
import numpy as np
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.misc import scandir
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedImageMixDataset(data.Dataset):
    """Paired image dataset with multi-root mixing by ratio.

    Required keys:
        dataroot_gt (str | list[str])
        dataroot_lq (str | list[str])
        mix_ratio (list[float], optional)
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = deepcopy(opt['io_backend'])
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        gt_roots = opt['dataroot_gt']
        lq_roots = opt['dataroot_lq']
        if isinstance(gt_roots, (str, bytes)):
            gt_roots = [gt_roots]
        if isinstance(lq_roots, (str, bytes)):
            lq_roots = [lq_roots]
        if len(gt_roots) != len(lq_roots):
            raise ValueError(f'dataroot_gt and dataroot_lq must have same length. Got {len(gt_roots)} vs {len(lq_roots)}')

        self.num_datasets = len(gt_roots)
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        mix_ratio = opt.get('mix_ratio', None)
        if mix_ratio is None:
            mix_ratio = [1.0] * self.num_datasets
        if len(mix_ratio) != self.num_datasets:
            raise ValueError(f'mix_ratio length must match datasets. Got {len(mix_ratio)} vs {self.num_datasets}')

        ratios = np.asarray(mix_ratio, dtype=np.float64)
        if np.any(ratios < 0) or ratios.sum() <= 0:
            raise ValueError('mix_ratio values must be non-negative and sum to > 0')
        self.mix_prob = ratios / ratios.sum()

        meta_info = opt.get('meta_info_file', None)
        recursive = opt.get('recursive', False)
        if self.io_backend_opt['type'] == 'lmdb':
            if self.num_datasets != 1:
                raise NotImplementedError('lmdb mode does not support multiple dataroots in PairedImageMixDataset')
            self.io_backend_opt['db_paths'] = [lq_roots[0], gt_roots[0]]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths_list = [paired_paths_from_lmdb([lq_roots[0], gt_roots[0]], ['lq', 'gt'])]
        elif meta_info is not None:
            if isinstance(meta_info, (list, tuple)):
                if len(meta_info) != self.num_datasets:
                    raise ValueError('meta_info_file list length must match datasets')
                self.paths_list = [
                    paired_paths_from_meta_info_file([lq, gt], ['lq', 'gt'], mi, self.filename_tmpl)
                    for lq, gt, mi in zip(lq_roots, gt_roots, meta_info)
                ]
            else:
                if self.num_datasets != 1:
                    raise ValueError('meta_info_file must be a list when using multiple dataroots')
                self.paths_list = [
                    paired_paths_from_meta_info_file([lq_roots[0], gt_roots[0]], ['lq', 'gt'], meta_info, self.filename_tmpl)
                ]
        else:
            if recursive:
                self.paths_list = [
                    self._paired_paths_from_folder_recursive(lq, gt, self.filename_tmpl)
                    for lq, gt in zip(lq_roots, gt_roots)
                ]
            else:
                self.paths_list = [
                    paired_paths_from_folder([lq, gt], ['lq', 'gt'], self.filename_tmpl)
                    for lq, gt in zip(lq_roots, gt_roots)
                ]

        self.lengths = [len(p) for p in self.paths_list]
        self.total_len = int(sum(self.lengths)) if self.lengths else 0

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        dataset_idx = int(np.random.choice(self.num_datasets, p=self.mix_prob))
        paths = self.paths_list[dataset_idx]
        path_idx = int(np.random.randint(len(paths)))
        gt_path = paths[path_idx]['gt_path']
        lq_path = paths[path_idx]['lq_path']

        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return self.total_len

    @staticmethod
    def _paired_paths_from_folder_recursive(input_folder, gt_folder, filename_tmpl):
        input_paths = list(scandir(input_folder, recursive=True, full_path=False))
        gt_paths = list(scandir(gt_folder, recursive=True, full_path=False))
        assert len(input_paths) == len(gt_paths), (
            f'lq and gt datasets have different number of images: {len(input_paths)}, {len(gt_paths)}.'
        )
        input_set = set(input_paths)
        paths = []
        for gt_rel in gt_paths:
            basename, ext = osp.splitext(osp.basename(gt_rel))
            input_name = f'{filename_tmpl.format(basename)}{ext}'
            rel_dir = osp.dirname(gt_rel)
            if rel_dir:
                input_rel = osp.join(rel_dir, input_name)
            else:
                input_rel = input_name
            assert input_rel in input_set, f'{input_rel} is not in lq_paths.'
            paths.append({'lq_path': osp.join(input_folder, input_rel), 'gt_path': osp.join(gt_folder, gt_rel)})
        return paths
