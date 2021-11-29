#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch, torchvision
from torch import nn
import torch.nn.functional as F
import numpy as np

# Self implemented
class MCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        error_norm = torch.mean(torch.sum(torch.sum(
            torch.square(torch.abs(X - Y)), dim=(-1, -2)),
                dim=-1))

        return error_norm

# Self implemented
class PSNRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        error_mse = torch.mean(torch.square(X - Y))
        self_max  = torch.max(torch.square(X))

        return error_mse / self_max


# Self implemented
class NMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        error_norm = torch.square(torch.norm(X - Y))
        self_norm  = torch.square(torch.norm(X))

        return error_norm / self_norm
        
# From fMRI
class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()

# Custom step scheduler
def lambda_step(drop_epochs, drop_factor):
    # Inner function trick
    def core_function(epoch):
        if np.isin(epoch, drop_epochs):
            return drop_factor
        else:
            return 1.

    return core_function




# following from this link https://learnopencv.com/paired-image-to-image-translation-pix2pix/
adversarial_loss = nn.BCEWithLogitsLoss() #nn.BCELoss() #wgan loss is also a good option
l1_loss = nn.L1Loss()


def generator_loss(generated_image, target_img, G, real_target, Lambda = 100):
    # G: Output predictions from the discriminator, when fed with generator-produced images.
    gen_loss = adversarial_loss(G, real_target)
    loss_val = l1_loss(generated_image, target_img)
    gen_total_loss = gen_loss + (Lambda* loss_val)
    # print(gen_loss)
    return gen_total_loss

def generator_loss_separately(generated_image, target_img, G, real_target, Lambda = 100):
    # G: Output predictions from the discriminator, when fed with generator-produced images.
    # similar to the above function, just it gives 2 outputs
    adv_loss = adversarial_loss(G, real_target)
    l1_l = l1_loss(generated_image, target_img)
    return l1_l, adv_loss

def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss

class NRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        error_norm = torch.square(torch.norm(X - Y))
        self_norm  = torch.square(torch.norm(X))

        return torch.sqrt(error_norm / self_norm)



''' Using VGGperceptualloss as used from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
    Might need to update this so as to make sure that it works with the current setup and with absolute value images
'''
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
