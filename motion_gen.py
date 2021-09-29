import h5py
import numpy as np
import sigpy as sp
import torch
from torch.utils.data import Dataset
import scipy
from scipy import ndimage

def adjoint_op(ksp,mask,maps):
        # Multiply input with mask and pad
        mask_adj_ext = mask
        ksp_padded  = ksp * mask_adj_ext

        # Get image representation of ksp
        img_ksp = sp.ifft(ksp_padded,axes=(- 2, - 1))
        # Pointwise complex multiply with complex conjugate maps
        mult_result = img_ksp * np.conj(maps)

        # Sum on coil axis
        x_adj = np.sum(mult_result, axis=0)

        return x_adj

def forward_op(img_kernel, maps, mask):
        # Pointwise complex multiply with maps
        mult_result = img_kernel * maps

        # Convert back to k-space
        result = sp.fft(mult_result,axes=(- 2, - 1))
        # Multiply with mask
        result = result * mask

        return result

def translate(input, delta_x, delta_y):
    output = np.roll(input, delta_x, axis = 1)
    output = np.roll(output, delta_y, axis = 0)
    return output

class MotionCorrupt(Dataset):
    def __init__(self, sample_list, maps_list, num_slices,
                 center_slice, num_shots):
        self.sample_list  = sample_list
        self.num_slices   = num_slices
        self.center_slice = center_slice
        self.maps         = maps_list # Pre-estimated sensitivity maps
        self.num_shots    = num_shots

    def __len__(self):
        return len(self.sample_list) * self.num_slices

    def __getitem__(self, idx):

        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Separate slice and sample
        sample_idx = idx // self.num_slices
        slice_idx  = self.center_slice + \
            np.mod(idx, self.num_slices) - self.num_slices // 2

        # Load MRI image
        with h5py.File(self.sample_list[sample_idx], 'r') as contents:
            # Get k-space for specific slice
            k_image = np.asarray(contents['kspace'][slice_idx])
            ref_rss = np.asarray(contents['reconstruction_rss'][slice_idx])

        # If desired, load external sensitivity maps
        if not self.maps is None:
            with h5py.File(self.maps[sample_idx], 'r') as contents:
                # Get sensitivity maps for specific slice
                s_maps      = np.asarray(contents['s_maps'][slice_idx])#was map_idx



        gt_img = adjoint_op(k_image, np.ones(1), s_maps)

        lines_per_shot = int(np.ceil(k_image.shape[-1]/self.num_shots)) #how many lines should be taken for each motion state

        #generate motion corrupted image
        for shot in range(self.num_shots):
            angle = np.random.uniform(low = -15.0, high = 15.0)
            delta_x = int(np.ceil(np.random.uniform(low = -50.0, high = 50.0)))
            delta_y = int(np.ceil(np.random.uniform(low = -50.0, high = 50.0)))
            motion_img = ndimage.rotate(input=gt_img, angle=angle,
                                        axes=(1, 0), reshape=False, output=None, order=3,
                                        mode='constant', cval=0.0, prefilter=True) #apply rotation

            motion_img = translate(input = motion_img, delta_x = delta_x, delta_y = delta_y)

            motion_ksp = sp.fft(motion_img,axes=(- 2, - 1), norm = 'ortho')

            start_line = shot*lines_per_shot
            end_line   = (shot+1)*lines_per_shot

            if shot == 0:
                lines = motion_ksp[:,start_line:end_line]
                ksp_motion_corrupt = lines
            if shot > 0:
                if shot<(self.num_shots-1):
                    lines = motion_ksp[:,start_line:end_line]
                else:
                    lines = motion_ksp[:,start_line:]

                ksp_motion_corrupt = np.append(ksp_motion_corrupt, lines,axis=1)


        img_motion_corrupt = sp.ifft(ksp_motion_corrupt,axes=(- 2, - 1))
        scale_gt = np.max(abs(gt_img))
        scale_motion = np.max(abs(img_motion_corrupt))

        img_motion_corrupt = img_motion_corrupt/scale_motion
        gt_img = gt_img/scale_gt
        data_range = np.max(abs(gt_img))
        sample = {'idx': idx,
                  'ksp_motion_corrupt': ksp_motion_corrupt.astype(np.complex64),
                  'img_motion_corrupt': img_motion_corrupt.astype(np.complex64),
                  'ksp_gt': k_image.astype(np.complex64),
                  'img_gt': gt_img.astype(np.complex64),
                  'data_range': data_range
#                   'acs_image': acs_image.astype(np.float32),
#                   'norm_const': norm_const.astype(np.float32)
                 }

        return sample
