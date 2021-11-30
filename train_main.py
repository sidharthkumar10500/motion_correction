#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import h5py, os, torch, glob, scipy
import numpy as np
import sigpy as sp
import torch.fft as torch_fft
from torch.utils.data import Dataset
from scipy import ndimage
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from Unet import Unet
from losses import SSIMLoss, MCLoss, NMSELoss, NRMSELoss
from torch.optim import Adam
from tqdm import tqdm
from datagen import MotionCorrupt
import argparse
import training_funcs


parser = argparse.ArgumentParser(description='Reading args for running the deep network training')
parser.add_argument('-e','--epochs', type=int, default=10, metavar='', help = 'number of epochs to train the network') #positional argument
parser.add_argument('-rs','--random_seed', type=int, default=80, metavar='', help = 'Random reed for the PRNGs of the training') #optional argument
parser.add_argument('-lr','--learn_rate', type=float, default=0.0001, metavar='', help = 'Learning rate for the network') #optional argument
parser.add_argument('-ma','--model_arc', type=str, default='GAN', metavar='',choices=['UNET', 'GAN'], help = 'Choose the type of network to learn')
parser.add_argument('-l','--loss_type', type=str, default='L1', metavar='',choices=['SSIM', 'L1', 'L2', 'Perc_L'], help = 'Choose the loss type for the main network')
parser.add_argument('-G','--GPU_idx',  type =int, default=2, metavar='',  help='GPU to Use')
parser.add_argument('-B','--batch_size',  type =int, default=10, metavar='',  help='Batch_size')
parser.add_argument('-lb','--Lambda', type=float, default=1,metavar='', help = 'variable to weight loss fn w.r.t adverserial loss')
parser.add_argument('-df','--data_file', type=str, default='mdme_data', metavar='',choices=['mdme_data', 'available_input_data'], help = 'Data on which the model need to be trained')
parser.add_argument('-de','--disc_epoch', type=int, default=1, metavar='', help = 'epochs for training the disc separately') 
parser.add_argument('-ge','--gen_epoch' , type=int, default=1, metavar='', help = 'epochs for training the gen separately')
parser.add_argument('-nw', '--num_workers' , type=int, default=2 , metavar='', help='number of workers to use')
parser.add_argument('-se', '--start_ep' , type=int, default=0 , metavar='', help='start epoch for training')
parser.add_argument('-ee', '--end_ep'   , type=int, default=200, metavar='', help='end epoch for training')
parser.add_argument('-ch', '--channels' , type=int, default=64 , metavar='', help='num channels for UNet')
parser.add_argument('-sc', '--scan_type', type=str, default='random_cart' , help='takes only random_cart, or alt_cart')

if __name__ == '__main__':
    args = parser.parse_args()
    args.step_size   = 10  # Number of epochs to decay lr with gamma in adam
    args.decay_gamma = 0.5
    print(args) #print the read arguments

    random_seed = args.random_seed  #changed to 80 to see the trianing behaviour on a different set
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # Disaster: trying to make the algorithms reproducible
    # torch.use_deterministic_algorithms(True) # if you want to set the use of determnistic algorihtms with all of pytorch, this have issues when using patch based with SSIM (that in itself is not a good idea to use anyway)
    torch.backends.cudnn.deterministic = True # Only affects convolution operations
    torch.backends.cudnn.benchmark     = False #if you want to replicate the results make this true

    # Make pytorch see the same order of GPUs as seen by the nvidia-smi command
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda:{}".format(args.GPU_idx) if torch.cuda.is_available() else "cpu")
    args.device = device
    # Global directory
    local_dir =  'train_results/model_%s_loss_type_%s'\
        %(args.model_arc, args.loss_type) 
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    args.local_dir = local_dir


    # Creating the dataloaders
    if args.scan_type == 'random_cart':
        new_dir = '/home/blevac/motion_cor_training_data/random_cart/'
        train_files_1 = sorted(glob.glob(new_dir + '/*.pt'))

    elif args.scan_type == 'alt_cart':
        new_dir = '/home/blevac/motion_cor_training_data/alternating_scan/'
        train_files_1 = sorted(glob.glob(new_dir + '/*.pt'))

    dataset = MotionCorrupt(sample_list = train_files_1, num_slices=10, center_slice = 7)
    train_loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    print('Training data length:- ',dataset.__len__())
    args.train_loader = train_loader
    args.val_loader = train_loader #for right now using same as the training, need to replace this with the validation dataset

    UNet1 = Unet(in_chans = 2, out_chans= 2,chans=args.channels).to(args.device)
    UNet1.train()
    # print('Number of parameters in the generator:- ', np.sum([np.prod(p.shape) for p in UNet1.parameters() if p.requires_grad]))

    import torchvision.models as models
    resnet = models.resnet18()
    Discriminator2 = resnet.to(device)

    args.generator     = UNet1
    args.discriminator = Discriminator2 #now using the vgg network as the discriminator
    if (args.model_arc == 'GAN'):
        training_funcs.GAN_training(args)
    elif(args.model_arc == 'UNET'):
        training_funcs.UNET_training(args)