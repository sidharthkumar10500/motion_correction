import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
import sigpy as sp
from scipy import ndimage
from tqdm import tqdm
import os

def motion_corrupt(gt_img):
    #input(H,W): clean Ground truth image you wish to add motion to
    #output(H,W): motion corrupted image

    num_shots = 7 # 7 for 'random', 2 for 'alternating'
    traj = 'cart'
    scan_order='random'

    lines_per_shot         = int(np.ceil(gt_img.shape[-1]/num_shots)) #how many lines should be taken for each motion state

    line_order             = np.arange(gt_img.shape[-1])
    rp_lines               = np.random.permutation(line_order)
    skip_lines             = [line_order[n::num_shots] for n in range(num_shots)]
    ksp_motion_corrupt     = np.zeros(gt_img.shape).astype(np.complex64)

    for shot in range(num_shots):
        #gnerate new motion state for each "shot"
        start_line = shot*lines_per_shot
        end_line   = (shot+1)*lines_per_shot

        angle = np.random.uniform(low = -2.0, high = 2.0) #pick random rotation
        delta_x = 0#int(np.ceil(np.random.uniform(low = -50.0, high = 50.0)))
        delta_y = 0#int(np.ceil(np.random.uniform(low = -50.0, high = 50.0)))

        motion_img = ndimage.rotate(input=gt_img, angle=angle,
                                axes=(1, 0), reshape=False, output=None, order=3,
                                mode='constant', cval=0.0, prefilter=True) #apply rotation


        motion_ksp = sp.fft(motion_img) #apply FFT to go from image space to k-space

        #select non-overlapping regions of k-space from each shot to save in one universal k-space
        if scan_order == 'random':
            ksp_motion_corrupt[:,rp_lines[start_line:end_line]] = motion_ksp[:,rp_lines[start_line:end_line]]
        elif scan_order == 'linear':
            ksp_motion_corrupt[:,line_order[start_line:end_line]] = motion_ksp[:,line_order[start_line:end_line]]
        elif scan_order == 'skip':
            ksp_motion_corrupt[:,skip_lines[shot]] = motion_ksp[:,skip_lines[shot]]
    #take IFFT of corrupt k-space to obtain corrupt image
    img_motion_corrupt = sp.ifft(ksp_motion_corrupt)
    return img_motion_corrupt



clean_data_folder = '/home/sidharth/sid_notebooks/motion_correction/val_data'
all_files = sorted(glob.glob(clean_data_folder + '/*.pt'))
new_dir = '/home/blevac/Junk/'

for file in tqdm(all_files):

    scan = torch.load(file)
    file_name = os.path.basename(file)
    gt_imgs = scan['mvue_recon'] #retrieve clean samples from each patient scan
    mo_co_full = np.zeros(gt_imgs.shape).astype(complex)

    for slice in range(gt_imgs.shape[-1]):
        #generate motion for all slices in each patient file
        mo_co_single = motion_corrupt(gt_img = gt_imgs[...,slice])
        mo_co_full[...,slice] = mo_co_single

    #save all motion corrupt images with ground truth images
    torch.save({
            'gt_imgs': gt_imgs,
            'moco_imgs': mo_co_full}, new_dir + file_name)
