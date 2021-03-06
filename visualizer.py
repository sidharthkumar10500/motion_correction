import numpy as np
import matplotlib.pyplot as plt
import torch, os, glob
from dotmap import DotMap
from Unet import Unet
from torchvision import transforms
from torch.utils.data import DataLoader
from datagen import MotionCorrupt
from losses import SSIMLoss, generator_loss, discriminator_loss, generator_loss_separately, adversarial_loss, NRMSELoss
import torchvision.models as models
vgg16 = models.vgg16()
"""
The aim of this file is to generate all the results for a particular trained model
"""

# go to the directory from which you want to load the pre-trained models
# os.chdir('/home/sidharth/sid_notebooks/UNET_GAN2_training/train_results/model_GAN_input_data_mdme_data_loss_type_L1_mode_Full_img/val_split_0.2_learning_rate_0.0001_epochs_10_lambda_1_gen_epoch_10_disc_epoch_20')
os.chdir('/home/sidharth/sid_notebooks/motion_correction/train_results/model_GAN_loss_type_L1/learning_rate_0.0001_epochs_5_lambda_1_gen_epoch_2_disc_epoch_2')

saved_results = torch.load('saved_weights.pt',map_location='cpu')#load the saved weights
hparams   =  saved_results['hparams']
hparams.device = 'cpu' #all gpus are clogged right now, use gpus if they are available
hparams.n_channels = 2
UNet1 = Unet(in_chans = hparams.n_channels, out_chans=hparams.n_channels,chans=hparams.channels).to(hparams.device)
UNet1.load_state_dict(saved_results['model_state_dict'])
UNet1.eval()
val_loader = hparams.val_loader
# create a new local_dir to save test results
# local_dir = hparams.global_dir + '/Test_images_learning_rate_{:.4f}_epochs_{}_lambda_{}'.format(hparams.lr,hparams.epochs,hparams.Lambda) 
local_dir = hparams.local_dir + '/Test_images_learning_rate_{:.4f}_epochs_{}_lambda_{}_{}_disc_epochs_{}'.format(hparams.learn_rate,hparams.epochs,hparams.Lambda,hparams.model_arc,hparams.disc_epoch) 

print(local_dir)
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

# Creating the dataloaders, change this to different test sets etc
if hparams.scan_type == 'random_cart':
    new_dir = '/home/blevac/motion_cor_val_data/random_scan/'
    train_files_1 = sorted(glob.glob(new_dir + '/*.pt'))

elif hparams.scan_type == 'alt_cart':
    new_dir = '/home/blevac/motion_cor_training_data/alternating_scan/'
    train_files_1 = sorted(glob.glob(new_dir + '/*.pt'))

dataset = MotionCorrupt(sample_list = train_files_1, num_slices=10, center_slice = 7)
train_loader  = DataLoader(dataset, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.num_workers, drop_last=True)
print('Training data length:- ',dataset.__len__())
hparams.train_loader = train_loader
hparams.val_loader = train_loader #for right now using same as the training, need to replace this with the validation dataset

SSIM       = SSIMLoss()
NRMSE      = NRMSELoss()
for index, sample in (enumerate(train_loader)):
    input_img            = torch.view_as_real(sample['img_motion_corrupt']).permute(0,3,1,2)
    model_out            = UNet1(input_img).permute(0,2,3,1)
    out                  = torch.view_as_complex(model_out.contiguous())
    generated_image      = torch.abs(out)
    target_img           = torch.abs(sample['img_gt'])

    NN_output = generated_image[0,:,:].cpu().detach().numpy().squeeze()
    actual_out = target_img[0,:,:].cpu().detach().numpy().squeeze()
    actual_in = torch.abs(sample['img_motion_corrupt'])
    actual_in = actual_in[0,:,:].cpu().detach().numpy().squeeze()
    nrmse_in  = NRMSE(torch.from_numpy(actual_out),torch.from_numpy(actual_in))
    nrmse_gan = NRMSE(torch.from_numpy(actual_out),torch.from_numpy(NN_output))
    SSIM_in   = SSIM(torch.from_numpy(actual_out[None,None,:,:]),torch.from_numpy(actual_in[None,None,:,:]) , torch.tensor([1]))
    SSIM_gan  = SSIM(torch.from_numpy(actual_out[None,None,:,:]),torch.from_numpy(NN_output[None,None,:,:]), torch.tensor([1]))
    plt.figure(figsize=(16,6))
    # plt.suptitle('test_image_index = {}'.format(index), fontsize=16)
    plt.subplot(1,4,1)
    plt.imshow(np.abs(actual_in),cmap='gray')#,vmax=0.5,vmin=0)
    plt.title('Input, NRMSE = {:.4f}, SSIM = {:.4f}'.format(nrmse_in,SSIM_in))
    plt.colorbar()
    plt.axis('off')
    plt.subplot(1,4,2)
    plt.imshow(np.abs(NN_output),cmap='gray')#,vmax=0.5,vmin=0)
    plt.title('Gen Out, NRMSE = {:.4f}, SSIM = {:.4f}'.format(nrmse_gan,SSIM_gan))
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1,4,3)
    plt.imshow(np.abs(actual_out),cmap='gray')#,vmax=0.5,vmin=0)
    plt.title('Ground Truth')
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1,4,4)
    plt.imshow(np.abs(NN_output-actual_out),cmap='gray')#,vmax=0.5*0.5,vmin=0)
    plt.title('Difference 2X')
    plt.axis('off')
    plt.colorbar()
        # Save
    plt.tight_layout()
    plt.savefig(local_dir + '/test_image_index = {}.png'.format(index), dpi=100)
    plt.close()