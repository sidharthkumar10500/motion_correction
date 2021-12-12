# motion_correction
This file is for implementing motion correction deep learning project for the comp vision class.

## Table of contents
* [Requirments](#Requirments)
* [Simulating_Motion](#Simulating_Motion)
* [UNET](#UNET)
* [GAN](#GAN)
* [Diffusion](#GAN)

## Requirments
```
$ pip install -r requirements.txt 
```
before running each method at inference download a subset of the validation dataset at: https://drive.google.com/drive/folders/16PskzHb4IJYeXGBnryjGSESgDx9dGfkE?usp=sharing. Make sure that you change the file paths in all the relevent locations in the code to ensure you get the data from the correct location on your machine.
## Simulating_Motion
run the following script with the correct input and output file paths on your PC to generate motion:

```
$ python motion_gen.py
```

## UNET

## GAN

## Diffusion
*Please note that most of the Diffusion repo posted here was from [1]. We adapted the dataloader to correctly take our data and altered the training function to save the relevant outputs.*

First you must download the saved diffusion model from our google drive (https://drive.google.com/drive/folders/16PskzHb4IJYeXGBnryjGSESgDx9dGfkE?usp=sharing). Place the folder *ncsv2-mri-mvue* inside the *Diffusion_Model* folder to maintain the assumed file structure.

To run the Diffusion model at inference time please alter the validation data file path in *aux_motion_data.py* for your specific machine. If you have generated your own motion corrupt data using *motion_gen.py* then the file path should be the same that you used to generate the images. Then use the following command:
```
$ python motion_inference.py --gpu 2 --anatomy brain --batch_size 1 --normalize_grad 1 --batches 0 9 --extra_accel 1 --noise_boost 1 --dc_boost 0.1 --contrast NA --val_num 0
```
[1] Ajil   Jalal,   Marius   Arvinte,   Giannis   Daras,   Eric   Price,Alexandros G Dimakis, and Jonathan I Tamir.  Robust compressed sensing mri with deep generative priors. Advances in Neural Information Processing Systems, 2021
