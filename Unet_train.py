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
from Unet import Unet
from losses import SSIMLoss, MCLoss, NMSELoss, NRMSELoss
from torch.optim import Adam
from tqdm import tqdm
from datagen import MotionCorrupt
from optparse import OptionParser
import argparse



def get_args():
    parser = OptionParser()
    parser.add_option('--pats'    , '--pats'    , type='int', default=1             , help = '# of patients')
    parser.add_option('--seed'    , '--seed'    , type='int', default=1500          , help='random seed to use')
    parser.add_option('--GPU'     , '--GPU'     , type='int', default=0             , help='GPU to Use')
    parser.add_option('--num_work', '--num_work', type='int', default=2             , help='number of workers to use')
    parser.add_option('--start_ep', '--start_ep', type='int', default=0             , help='start epoch for training')
    parser.add_option('--end_ep'  , '--end_ep'  , type='int', default=200           , help='end epoch for training')
    parser.add_option('--ch'      , '--ch'      , type='int', default=32            , help='num channels for UNet')
    parser.add_option('--scan'    , '--scan'    , type='str', default='random_cart' , help='takes only random_cart, or alt_cart')
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
    print(args)

    GPU_ID                = args.GPU
    global_seed           = args.seed
    num_workers           = args.num_work
    start_epoch           = args.start_ep
    end_epoch             = args.end_ep
    ch                    = args.ch
    scan_type             = args.scan

    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    # Fix seed
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)


    if scan_type == 'random_cart':
        new_dir = '/home/blevac/motion_cor_training_data/random_cart/'
        train_files_1 = sorted(glob.glob(new_dir + '/*.pt'))

    elif scan_type == 'alt_cart':
        new_dir = '/home/blevac/motion_cor_training_data/alternating_scan/'
        train_files_1 = sorted(glob.glob(new_dir + '/*.pt'))

    dataset = MotionCorrupt(sample_list = train_files_1, num_slices=10, center_slice = 7)
    train_loader  = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=num_workers, drop_last=True)


    model      = Unet(in_chans = 2 , out_chans = 2, chans = 32,
                                num_pool_layers = 4, drop_prob = 0.0)
    model      = model.cuda()

    optimizer  = Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, 10,
                       gamma=0.5)

    ssim       = SSIMLoss().cuda()
    nrmse_loss = NRMSELoss()


    running_training  = 0.0
    running_nrmse     = 0.0
    running_ssim      = -1.0


    local_dir = 'Results/%s/seed%d' % (scan_type, global_seed )
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)


    if start_epoch>0:
        saved_model = torch.load(local_dir + '/epoch'+str(start_epoch-1)+'_last_weights.pt')
        model.load_state_dict(saved_model['model_state_dict'])
        optimizer.load_state_dict(saved_model['optimizer'])



    for epoch_idx in range(start_epoch, end_epoch):
        for sample_idx, sample in tqdm(enumerate(train_loader)):

            # Move to CUDA
            for key in sample.keys():
                try:
                    sample[key] = sample[key].cuda()
                except:
                    pass

            inp_img        = torch.view_as_real(sample['img_motion_corrupt']).permute(0,3,1,2)
            model_out      = model(inp_img).permute(0,2,3,1)
            out            = torch.view_as_complex(model_out.contiguous())
            est_img_rss    = torch.abs(out)

            # gt_img         = torch.abs(sample['img_gt'])
            gt_img         = torch.abs(sample['img_gt'])
            data_range     = sample['data_range']
            ssim_loss      = ssim(est_img_rss[:,None], gt_img[:,None], data_range)

            loss = ssim_loss
            # Other Loss for tracking
            with torch.no_grad():
                nrmse      = nrmse_loss(gt_img,est_img_rss)

            # if loss<0.1:
            #     plt.figure(figsize=(12,8))
            #     plt.subplot(1,4,1)
            #     plt.title('GT Image')
            #     plt.imshow(gt_img[0,...].cpu(), cmap='gray')
            #     plt.axis('off')
            #
            #     plt.subplot(1,4,2)
            #     title1 = 'Alt GT, SSIM:%.2f' %(1-ssim_loss_gt_alt.cpu().detach().numpy())
            #     plt.title(title1)
            #     plt.imshow(abs(sample['altered_gt'])[0,...].cpu(), cmap = 'gray')
            #     plt.axis('off')
            #
            #     plt.subplot(1,4,3)
            #     title2 = 'Motion Image, SSIM:%.2f' %(1-ssim_loss_motion.cpu().detach().numpy())
            #     plt.title(title2)
            #     plt.imshow(abs(sample['img_motion_corrupt'])[0,...].cpu(), cmap = 'gray')
            #     plt.axis('off')
            #
            #     plt.subplot(1,4,4)
            #     title3 = 'Corr Image, SSIM:%.2f' %(1-ssim_loss_t.cpu().detach().numpy())
            #     plt.title(title3)
            #     plt.imshow(est_img_rss[0,...].cpu().detach().numpy(), cmap = 'gray')
            #     plt.axis('off')
            #
            #     # Save
            #     plt.tight_layout()
            #     plt.savefig(local_dir + '/good_correction_samples_epoch%d.png' % epoch_idx, dpi=300)
            #     plt.close()

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

        scheduler.step()
