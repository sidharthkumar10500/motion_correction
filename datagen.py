import numpy as np
import sigpy as sp
import torch, os
from torch.utils.data import Dataset
import scipy
from scipy import ndimage
"""
Dataloader for the motion corrupt images 
"""
class MotionCorrupt_SE(Dataset):
    def __init__(self, root_dir):
        self.root_dir  = root_dir #location of the text file
        with open(root_dir, "r") as text_file:
            files = text_file.readlines()
            files = [file.rstrip() for file in files]
        self.datafiles = files

    def __len__(self):
        return len(self.datafiles)
        
    def __getitem__(self, idx):
        datafile = self.datafiles[idx]
        os.chdir('/home/sidharth/sid_notebooks/')#change current working directory
        loaded_data = torch.load(datafile)
        local_X = np.asarray(loaded_data['SE_img'])
        local_y = np.asarray(loaded_data['GT_img'])
        scale2 = np.max(abs(local_y))
        scale1 = np.max(abs(local_X))

        gt_img = local_y/scale2
        motion_img = local_X/scale1

        sample = {'idx': idx,
                  'img_motion_corrupt': motion_img.astype(np.complex64),
                  'img_gt': gt_img.astype(np.complex64),
                  'data_range': 1.0
             }
        return sample

class MotionCorrupt_FSE(Dataset):
    def __init__(self, root_dir):
        self.root_dir  = root_dir #location of the text file
        with open(root_dir, "r") as text_file:
            files = text_file.readlines()
            files = [file.rstrip() for file in files]
        self.datafiles = files

    def __len__(self):
        return len(self.datafiles)
        
    def __getitem__(self, idx):
        datafile = self.datafiles[idx]
        os.chdir('/home/sidharth/sid_notebooks/')#change current working directory
        loaded_data = torch.load(datafile)
        local_X = np.asarray(loaded_data['FSE_img'])
        local_y = np.asarray(loaded_data['GT_img'])
        scale2 = np.max(abs(local_y))
        scale1 = np.max(abs(local_X))

        gt_img = local_y/scale2
        motion_img = local_X/scale1

        sample = {'idx': idx,
                  'img_motion_corrupt': motion_img.astype(np.complex64),
                  'img_gt': gt_img.astype(np.complex64),
                  'data_range': 1.0
             }
        return sample