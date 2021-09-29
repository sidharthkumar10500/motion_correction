import h5py, os
import numpy as np
import sigpy as sp
import torch
import torch.fft as torch_fft
from torch.utils.data import Dataset
import scipy
from scipy import ndimage
import numpy as np
import glob
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from unet_branch import OldUnetModel
from losses import SSIMLoss, MCLoss, NMSELoss, NRMSELoss
from torch.optim import Adam
from tqdm import tqdm
from motion_gen import adjoint_op, forward_op, MotionCorrupt


os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# Fix seed
global_seed = 3000
torch.manual_seed(global_seed)
np.random.seed(global_seed)

# core_dir_1    = '/csiNAS/mridata/fastmri_brain/brain_multicoil_train/multicoil_train'
# maps_dir_1    = '/csiNAS/mridata/fastmri_brain/multicoil_train_espiritWc0_mvue_ALL'

core_dir_1    = '/csiNAS/brett/sites/site5/ksp'
maps_dir_1    = '/csiNAS/brett/sites/site5/maps'

train_files_1 = sorted(glob.glob(core_dir_1 + '/*.h5'))
train_maps_1  = sorted(glob.glob(maps_dir_1 + '/*.h5'))

train_files_1 = [train_files_1[idx] for idx in range(100)]
train_maps_1 = [train_maps_1[idx] for idx in range(100)]
dataset = MotionCorrupt(sample_list = train_files_1, maps_list=train_maps_1, num_slices=10, center_slice = 7, num_shots = 11)
train_loader  = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=20, drop_last=True)


num_epochs = 100

model      = OldUnetModel(in_chans = 2 , out_chans = 2, chans = 32,
                            num_pool_layers = 4, drop_prob = 0.0)
model      = model.cuda()

optimizer  = Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, 20,
                   gamma=0.5)

ssim       = SSIMLoss().cuda()
nrmse_loss = NRMSELoss()


running_training  = 0.0
running_nrmse     = 0.0
running_ssim      = -1.0

local_dir = 'Results/seed%d' % ( global_seed )
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

for epoch_idx in range(num_epochs):
    for sample_idx, sample in tqdm(enumerate(train_loader)):

        # Move to CUDA
        for key in sample.keys():
            try:
                sample[key] = sample[key].cuda()
            except:
                pass

        inp_img        = torch.view_as_real(sample['img_motion_corrupt']).permute(0,3,1,2)
        model_out      = model(input = inp_img).permute(0,2,3,1)
        out            = torch.view_as_complex(model_out.contiguous())
        est_img_rss    = torch.abs(out)

        gt_img         = torch.abs(sample['img_gt'])
        data_range     = sample['data_range']
        ssim_loss      = ssim(est_img_rss[:,None], gt_img[:,None], data_range)

        loss = ssim_loss
#         print("input max:", torch.max(abs(sample['img_motion_corrupt'])))
#         print("est_img max:", torch.max(est_img_rss))
#         print("gt_img max:", torch.max(gt_img))
#         print("data range:",data_range)
        # Other Loss for tracking
        with torch.no_grad():
            nrmse      = nrmse_loss(gt_img,est_img_rss)


        running_ssim     = 0.99 * running_ssim + 0.01 * (1-ssim_loss.item()) if running_ssim > -1. else (1-ssim_loss.item())
        running_nrmse    = 0.99 * running_nrmse + 0.01 * nrmse.item() if running_nrmse > 0. else nrmse.item()
        running_training = 0.99 * running_training + 0.01 * loss.item() if running_training > 0. else loss.item()
#         if (epoch_idx ==0) and (sample_idx == 0):
#             plt.figure()
#             plt.subplot(1,3,1)
#             plt.imshow(abs(gt_img[0,...].cpu()))
#             plt.subplot(1,3,2)
#             plt.imshow(abs(est_img_rss[0,...].detach().cpu()))
#             plt.subplot(1,3,3)
#             plt.imshow(abs(sample['img_motion_corrupt'][0,...].cpu()))
        # Backprop
        optimizer.zero_grad()
        ssim_loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), hparams.grad_clip)
        optimizer.step()

        print('Epoch %d ,Step %d, Batch loss %.4f. Avg. SSIM %.4f, Avg. NRMSE %.4f' % (
                    epoch_idx, sample_idx, loss.item(), running_ssim, running_nrmse))

        torch.save({
            'epoch': epoch_idx,
            'sample_idx': sample_idx,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()}, local_dir + '/epoch'+str(epoch_idx)+'_last_weights.pt')
 

    scheduler.step()
