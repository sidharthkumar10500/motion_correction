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
def get_data(anatomy):
    # Outputs
    global_ksp, global_maps  = [], []
    global_gt, global_slices = [], []
    global_filenames = []
    
    if anatomy == 'brain':
        global_files = sorted(glob.glob('./data_for_marius/*.h5'))
        target_files = [global_files[0], global_files[1],
                        global_files[4], global_files[5]]
        # For each file
        for file_idx, target_file in enumerate(target_files):
            with h5py.File(target_file, 'r') as contents:
                # Fetch stuff
                ksp  = np.asarray(contents['ksp'])
                maps = np.asarray(contents['maps'])
                
            # Estimate or preload "GT"
            gt_file = os.path.join(
                    './data_for_marius/estimated_gt', 
                    os.path.basename(target_file))
            if not os.path.exists(gt_file):
                gt_x = np.zeros((ksp.shape[-1], ksp.shape[0], ksp.shape[1]),
                                dtype=np.complex64)
                # Estimate each slice
                for slice_idx in tqdm(range(ksp.shape[-1])):
                    local_ksp  = ksp[..., slice_idx][None, :]
                    local_maps = maps[..., slice_idx][None, :]
                    # Run recon
                    local_gt = bart(1, 'pics -l2 -r 0.005 -i 100 -S -d 0 -e',
                        local_ksp, local_maps).squeeze()
                    # Store result
                    gt_x[slice_idx] = np.copy(local_gt)
                    
                # Save to file
                with h5py.File(gt_file, 'w') as hf:
                    hf.create_dataset('gt_x', data=gt_x)
                
            else:
                # Read from file
                with h5py.File(gt_file, 'r') as contents:
                    gt_x = np.asarray(contents['gt_x'])
                    
            # Reshape to [B, C, H, W]
            ksp  = np.transpose(ksp, (-1, -2, 0, 1))
            maps = np.transpose(maps, (-1, -2, 0, 1))
            
            # Add to global collections
            global_ksp.append(ksp)
            global_maps.append(maps)
            global_gt.append(gt_x)
            global_slices.append(ksp.shape[0])
            global_filenames.append(target_file)
            
        # Convert to arrays
        global_ksp  = np.vstack(global_ksp)
        global_maps = np.vstack(global_maps)
        global_gt   = np.vstack(global_gt)
            
    elif anatomy == 'knee':
        # Where is the data and the maps
        raw_dir = '/csiNAS2/slow/mridata/fastmri_knee/multicoil_val'
        map_dir = '/csiNAS2/slow/mridata/fastmri_knee/knee_multicoil_val_espiritWc0_maps'
        
        # Ajil's file list
        contents = torch.load('fs_files.pt')
        # Get core files and slices
        all_files = contents['fs_files']
        
        # Fetch data
        for file in all_files:
            core, num = os.path.basename(file).rsplit('_', 1)
            global_filenames.append(core + '.h5')
            global_slices.append(int(num[:-3]))
            with h5py.File(os.path.join(map_dir, global_filenames[-1]), 'r') as contents:
                global_maps.append(np.asarray(contents['s_maps'][global_slices[-1]]))
            # Get raw data
            with h5py.File(os.path.join(raw_dir, global_filenames[-1]), 'r') as contents:
                local_ksp = np.asarray(contents['kspace'][global_slices[-1]])
                if True:
                    # Compute sum-energy of lines
                    line_energy = np.sum(np.square(np.abs(local_ksp)),
                                          axis=(0, 1))
                    dead_lines  = np.where(line_energy < 1e-16)[0] # Sufficient for FP32
                    # Always remove an even number of lines
                    dead_lines_front = np.sum(dead_lines < local_ksp.shape[-1]//2)
                    dead_lines_back  = np.sum(dead_lines > local_ksp.shape[-1]//2)
                    if np.mod(dead_lines_front, 2):
                        dead_lines = np.delete(dead_lines, 0)
                    if np.mod(dead_lines_back, 2):
                        dead_lines = np.delete(dead_lines, -1)
                    # Remove dead lines completely
                    local_ksp = np.delete(local_ksp, dead_lines, axis=-1)
                global_ksp.append(local_ksp)
            # Estimate MVUE
            local_mvue = np.sum(sp.ifft(global_ksp[-1], axes=(-1, -2)) *
                                np.conj(global_maps[-1]), axis=0)
            global_gt.append(sp.resize(local_mvue, (320, 320)))
            
    elif anatomy == 'fastmri_brain':
        # Where is the data and the maps
        raw_dir = '/csiNAS/mridata/fastmri_brain/brain_multicoil_val/multicoil_val/multicoil_val'
        map_dir = '/csiNAS/mridata/fastmri_brain/multicoil_val_espiritWc0_mvue_ALL'
        
        # Where is Ajil's storage
        lookup_dir = '/home/ajil/work/mri-stylegan/ajil_datasets/brains/mvue'
        
        # Get core files and slices
        all_files = sorted(glob.glob(lookup_dir + '/*.h5'))
        
        # Populate them
        for file in all_files:
            core, num = os.path.basename(file).rsplit('_', 1)
            global_filenames.append(core + '.h5')
            global_slices.append(int(num[:-3]))
            with h5py.File(os.path.join(map_dir, global_filenames[-1]), 'r') as contents:
                global_maps.append(np.asarray(contents['s_maps'][global_slices[-1]]))
            # Get raw data
            with h5py.File(os.path.join(raw_dir, global_filenames[-1]), 'r') as contents:
                global_ksp.append(np.asarray(contents['kspace'][global_slices[-1]]))
            # Estimate MVUE
            local_mvue = np.sum(sp.ifft(global_ksp[-1], axes=(-1, -2)) *
                                np.conj(global_maps[-1]), axis=0)
            global_gt.append(sp.resize(local_mvue, (384, 384)))
        
    # Return stuff
    return global_ksp, global_maps, global_gt, global_slices, global_filenames

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