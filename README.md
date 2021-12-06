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
To run the Diffusion model at inference time use the following command:
```
$ python motion_inference.py --gpu 1 --anatomy brain --batch_size 1 --normalize_grad 1 --batches 0 9 --extra_accel 1 --noise_boost 1 --dc_boost 1.0 --contrast T2
```
