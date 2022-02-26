#!/bin/bash
# bash script to run multiple training simultaneously and then choosing the best hyper parameters
# for model_type in GAN UNET
# do
#     for loss in  L1 SSIM # L2 #
#     do
#         python train_main.py -e 10 -ma $model_type -G 1 -de 1 -ge 1 -B 1 -lb_b 10
#     done
# done
# echo "All done"

# python train_main.py -e 10 -ma GAN -G 1 -de 20 -ge 10 -B 20 -lb 0.1 -lb_b 1 -sc FSE
python train_main.py -e 10 -ma GAN -G 1 -de 20 -ge 10 -B 20 -lb 0.1 -lb_b 1 -sc SE