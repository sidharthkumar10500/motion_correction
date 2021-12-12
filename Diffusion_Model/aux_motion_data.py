#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 17:10:34 2021

@author: yanni
"""

import os, glob, h5py, torch
import numpy as np
import sigpy as sp
from tqdm import tqdm

from bart import bart
from torch.utils.data import Dataset

# Plain, single-shot load
def get_motion_data(anatomy, contrast, val_num):
    # Outputs
    global_ksp, global_maps  = [], []
    global_gt, global_slices, global_motion = [], [], []
    global_filenames = []
    scan_type = 'random_cart'
    if anatomy == 'brain':
        # where to grab the data from
        if scan_type == 'random_cart':
            new_dir = '/home/blevac/motion_cor_val_data/random_scan/'
            global_files = sorted(glob.glob(new_dir + '/*.pt'))

        elif scan_type == 'alt_cart':
            new_dir = '/home/blevac/motion_cor_val_data/alternating_scan/'
            gobal_files = sorted(glob.glob(new_dir + '/*.pt'))
        if contrast == 'T2':
            idx = -20
        elif contrast == 'FLAIR':
            idx = 0
        elif contrast == 'NA':
            idx = val_num

        target_files = [global_files[idx]]

        # For each file
        for file_idx, target_file in enumerate(target_files):
            gt_img = torch.load(target_file)['gt_imgs'] #[H,W,B]
            motion_img = torch.load(target_file)['moco_imgs'] #[H,W,B]
            motion_ksp = sp.fft(motion_img, axes=(0,1), norm='ortho')
            # print('motion ksp shape:', motion_ksp.shape)
            # print('motion image shape:', motion_img.shape)
            m_img = np.transpose(motion_img, (-1,0,1))
            gt_x = np.zeros((motion_ksp.shape[-1], motion_ksp.shape[0], motion_ksp.shape[1]),
                            dtype=np.complex64)
            gt_x = np.transpose(gt_img, (-1,0,1))
            print('gt_x shape',gt_x.shape)
            # Reshape to [B, C, H, W]
            ksp  = np.transpose(motion_ksp, (-1, 0, 1))[:,None,...]
            maps = np.ones(ksp.shape)

            # Add to global collections
            global_ksp.append(ksp)
            global_maps.append(maps)
            global_gt.append(gt_x)
            global_slices.append(ksp.shape[0])
            global_filenames.append(target_file)
            global_motion.append(m_img)
        # Convert to arrays
        global_ksp  = np.vstack(global_ksp)
        global_maps = np.vstack(global_maps)
        global_gt   = np.vstack(global_gt)
        global_motion = np.vstack(global_motion)
    # Return stuff
    return global_ksp, global_maps, global_gt, global_slices, global_filenames, global_motion

class TruncatedMVUE(Dataset):
    def __init__(self, mvue_list,
                 raw_list,
                 R=1, pattern='random'):
        # Attributes
        self.mvue_list    = mvue_list
        self.raw_list     = raw_list
        self.R            = R
        self.pattern      = pattern

        # Access meta-data of each scan to get number of slices
        self.num_slices = np.zeros((len(self.mvue_list,)), dtype=int)
        for idx, file in tqdm(enumerate(self.mvue_list)):
            with h5py.File(file, 'r') as data:
                self.num_slices[idx] = int(data['mvue'].shape[0])

        # Create cumulative index for mapping
        self.slice_mapper = np.cumsum(self.num_slices) - 1 # Counts from '0'

    def __len__(self):
        return int(np.sum(self.num_slices)) # Total number of slices from all scans

    # Phase encode random mask generator
    def _get_mask(self, acs_lines=30, total_lines=384, R=1, pattern='random'):
        # Overall sampling budget
        num_sampled_lines = np.floor(total_lines / R)

        # Get locations of ACS lines
        # !!! Assumes k-space is even sized and centered, true for fastMRI
        center_line_idx = np.arange((total_lines - acs_lines) // 2,
                             (total_lines + acs_lines) // 2)

        # Find remaining candidates
        outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)

        if pattern == 'random':
            # Sample remaining lines from outside the ACS at random
            # np.random.seed(1)
            random_line_idx = np.random.choice(outer_line_idx,
                       size=int(num_sampled_lines - acs_lines), replace=False)
        elif pattern == 'equispaced':
            # Sample equispaced lines
            # !!! Only supports integer for now
            random_line_idx = outer_line_idx[::int(R)]

        # Create a mask and place ones at the right locations
        mask = np.zeros((total_lines))
        mask[center_line_idx] = 1.
        mask[random_line_idx] = 1.

        return mask

    # Cropping utility - works with numpy / tensors
    def _crop(self, x, wout, hout):
        w, h = x.shape[-2:]
        x1 = int(np.ceil((w - wout) / 2.))
        y1 = int(np.ceil((h - hout) / 2.))

        return x[..., x1:x1+wout, y1:y1+hout]

    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get scan and slice index
        # First scan for which index is in the valid cumulative range
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Offset from cumulative range
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)

        # Load raw data for specific scan and slice
        raw_file = os.path.join(self.raw_list[scan_idx])
        with h5py.File(raw_file, 'r') as data:
            # Get raw k-space
            kspace = np.asarray(data['kspace'][slice_idx])

        # Crop extra lines and reduce FoV by half in readout
        kspace = sp.resize(kspace, (
            kspace.shape[0], kspace.shape[1], 384))

        # Reduce FoV by half in the readout direction
        kspace = sp.ifft(kspace, axes=(-2,))
        kspace = sp.resize(kspace, (kspace.shape[0], 384,
                                    kspace.shape[2]))
        kspace = sp.fft(kspace, axes=(-2,)) # Back to k-space

        # !!! Removed ACS-based scaling if handled on the outside
        scale_factor = 1.

        # Scale data
        kspace = kspace / scale_factor

        # Get ground truth MVUE with ESPiRiT maps
        with h5py.File(self.mvue_list[scan_idx], 'r') as data:
            gt_mvue = np.asarray(data['mvue'][slice_idx])

        # Compute ACS size based on R factor and sample size
        total_lines = kspace.shape[-1]
        if 1 < self.R <= 6:
            # Keep 8% of center samples
            acs_lines = np.floor(0.08 * total_lines).astype(int)
        else:
            # Keep 4% of center samples
            acs_lines = np.floor(0.04 * total_lines).astype(int)

        # Get a mask
        mask = self._get_mask(acs_lines, total_lines,
                              self.R, self.pattern)

        # Mask k-space
        gt_ksp  = np.copy(kspace)
        kspace *= mask

        # Convert to reals
        kspace = np.stack((
            np.real(kspace),
            np.imag(kspace)), axis=-1)

        # Output
        sample = {
                  'ksp': kspace,
                  'mask': mask,
                  'gt_mvue': gt_mvue,
                  'gt_ksp': gt_ksp,
                  'scale_factor': scale_factor,
                  # Just for feedback
                  'scan_idx': scan_idx,
                  'slice_idx': slice_idx}

        return sample
