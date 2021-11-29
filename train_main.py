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


parser = argparse.ArgumentParser(description='Reading args for running the deep network training')
parser.add_argument('-e','--epochs', type=int, default=100, metavar='', help = 'number of epochs to train the network') #positional argument
parser.add_argument('-rs','--random_seed', type=int, default=80, metavar='', help = 'Random reed for the PRNGs of the training') #optional argument
parser.add_argument('-lr','--learn_rate', type=float, default=0.0001, metavar='', help = 'Learning rate for the network') #optional argument
parser.add_argument('-ma','--model_arc', type=str, default='GAN', metavar='',choices=['UNET', 'GAN'], help = 'Choose the type of network to learn')
parser.add_argument('-mm','--model_mode', type=str, default='Full_img', metavar='',choices=['Full_img', 'Patch'], help = 'Choose the mode to train the network either pass full image or patches')
parser.add_argument('-l','--loss_type', type=str, default='Perc_L', metavar='',choices=['SSIM', 'L1', 'L2', 'Perc_L'], help = 'Choose the loss type for the main network')
parser.add_argument('-G','--GPU_idx',  type =int, default=2, metavar='',  help='GPU to Use')
parser.add_argument('-lb','--Lambda', type=float, default=1,metavar='', help = 'variable to weight loss fn w.r.t adverserial loss')
parser.add_argument('-df','--data_file', type=str, default='mdme_data', metavar='',choices=['mdme_data', 'available_input_data'], help = 'Data on which the model need to be trained')
parser.add_argument('-de','--disc_epoch', type=int, default=10, metavar='', help = 'epochs for training the disc separately') 
parser.add_argument('-ge','--gen_epoch', type=int, default=10, metavar='', help = 'epochs for training the gen separately')
parser.add_argument('--num_work', '--num_work', type= int, default=2             , help='number of workers to use')
parser.add_argument('--start_ep', '--start_ep', type=int, default=0             , help='start epoch for training')
parser.add_argument('--end_ep'  , '--end_ep'  , type=int, default=200           , help='end epoch for training')
parser.add_argument('--ch'      , '--ch'      , type=int, default=32            , help='num channels for UNet')
parser.add_argument('--scan'    , '--scan'    , type=str, default='random_cart' , help='takes only random_cart, or alt_cart')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args) #print the read arguments
