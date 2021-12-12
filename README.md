# MRI Motion Correction
This repository hosts code for implementing motion correction deep learning project for the adv. topics in comp vision (deep learning) class.

## Table of contents
* [Requirements](#Requirements)
* [Simulating_Motion](#Simulating_Motion)
* [U-Net](#U-Net)
* [GAN](#GAN)
* [Diffusion](#GAN)

## Requirements
Download all the required packages in a new conda enviornment. 
```
$ pip install -r requirements.txt 
```
You should have a GPU of sufficient memory if you want to run the training/evaluation in batches, for lower memory GPUs run using a batch size of 1. 

To run inference on the pre-trained methods download a subet of the validation dataset at: https://drive.google.com/drive/folders/16PskzHb4IJYeXGBnryjGSESgDx9dGfkE?usp=sharing. Make sure to change the file paths in all the relevent locations in the code to ensure the correct location of data on your machine. 

## Simulating_Motion
To start from the fast_mri dataset, first download the dataset from the original fast_mri dataset (<https://fastmri.med.nyu.edu/>). Beware the dataset is quite large around ~2TB as it contains the raw kspace data for multiple coil acquisition. To generate the MVUE single coil images, run the following script:
```
python fastmri_convertor.py
```
Remember to change the location of fast_mri dataset on your machine. 
To generate the motion corrupt images run the following script with the correct input and output file paths on your machine:
```
$ python motion_gen.py
```

## U-Net
If you want to train either UNET or GAN from scratch, run the following bash script with selection of appropriate hyper-parameters:
```
bash bash_training_jobs.sh
```
This will run the training for different hyper-parameters and chooose the ones which give the best performance. 
If you just want to do inference, then download the pre-trained U-Net model parameters from the following drive (<https://drive.google.com/drive/folders/1W5BYIJeQLlEkPHhLCQmOJYlm2LWSN1Ie?usp=sharing>).
## GAN
To do GAN training from scratch, use the same script as was provided for U-Net, just change the appropriate flags. 
If you just want to do inference, then download the pre-trained GAN model parameters from the following drive (<https://drive.google.com/drive/folders/1QO2i3CF5iduM4COoxQZykhbDOBYkUIMv?usp=sharing>).
## Diffusion
*Please note that most of the Diffusion repo posted here was from [1]. We adapted the dataloader to correctly take our data and altered the training function to save the relevant outputs.*

First you must download the saved diffusion model from our google drive (https://drive.google.com/drive/folders/16PskzHb4IJYeXGBnryjGSESgDx9dGfkE?usp=sharing). Place the folder *ncsv2-mri-mvue* inside the *Diffusion_Model* folder to maintain the assumed file structure.

To run the Diffusion model at inference time please alter the validation data file path in *aux_motion_data.py* for your specific machine. If you have generated your own motion corrupt data using *motion_gen.py* then the file path should be the same that you used to generate the images. Then use the following command:
```
$ python motion_inference.py --gpu 2 --anatomy brain --batch_size 1 --normalize_grad 1 --batches 0 9 --extra_accel 1 --noise_boost 1 --dc_boost 0.1 --contrast NA --val_num 0
```
[1] Ajil   Jalal,   Marius   Arvinte,   Giannis   Daras,   Eric   Price,Alexandros G Dimakis, and Jonathan I Tamir.  Robust compressed sensing mri with deep generative priors. Advances in Neural Information Processing Systems, 2021
