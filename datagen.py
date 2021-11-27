
import numpy as np
import sigpy as sp
import torch
from torch.utils.data import Dataset
import scipy
from scipy import ndimage

class MotionCorrupt(Dataset):
    def __init__(self, sample_list, num_slices,
                 center_slice):
        self.sample_list  = sample_list #list of all of those pytorch files
        self.num_slices   = num_slices
        self.center_slice = center_slice

    def __len__(self):
        return len(self.sample_list) * self.num_slices

    def __getitem__(self, idx):

        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Separate slice and sample
        sample_idx = idx // self.num_slices
        slice_idx  = self.center_slice + np.mod(idx, self.num_slices) - self.num_slices // 2

        # Load MRI image
        gt_img = torch.load(self.sample_list[sample_idx])['gt_imgs'][...,slice_idx]
        motion_img = torch.load(self.sample_list[sample_idx])['moco_imgs'][...,slice_idx]
        scale1 = np.max(abs(gt_img))
        scale2 = np.max(abs(motion_img))
        gt_img = gt_img/scale1
        motion_img = motion_img/scale2
        sample = {'idx': idx,
                  'img_motion_corrupt': motion_img.astype(np.complex64),
                  'img_gt': gt_img.astype(np.complex64),
                  'data_range': 1.0
             }

        return sample
