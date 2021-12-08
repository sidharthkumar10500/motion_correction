# motion_correction
This file is for implementing motion correction deep learning project for the comp vision class.

## Table of contents
* [Requirments](#Requirments)
* [Simulating_Motion](#Simulating_Motion)
* [UNET](#UNET)
* [GAN](#GAN)
* [Diffusion](#GAN)

## Requirments


## Simulating_Motion
run the following script with the correct input and output file paths on your PC to generate motion:

```
$ python motion_gen.py
```

## UNET

## GAN

## Diffusion
Note that most of the Diffusion repo posted here was from [1]. We adapted the dataloader to correctly take our data and altered the training function to save the relevant outpututs.

To run the Diffusion model at inference time use the following command:
```
$ python motion_inference.py --gpu 1 --anatomy brain --batch_size 1 --normalize_grad 1 --batches 0 9 --extra_accel 1 --noise_boost 1 --dc_boost 1.0 --contrast T2
```
[1] Ajil   Jalal,   Marius   Arvinte,   Giannis   Daras,   Eric   Price,Alexandros G Dimakis, and Jonathan I Tamir.  Robust com-pressed sensing mri with deep generative priors.Advancesin Neural Information Processing Systems, 2021
