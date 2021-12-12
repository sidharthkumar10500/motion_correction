#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 18:26:07 2021

@author: yanni

Edited on 20th by Brett Levac
"""

import sys
sys.path.insert(0, './bart-0.6.00/python')
sys.path.append('./bart-0.6.00/python')

import torch, os, argparse
import numpy as np
import sigpy as sp
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOOLBOX_PATH"]    = './bart-0.6.00'

from dotmap import DotMap
from ncsnv2.models.ncsnv2 import NCSNv2Deepest

from utils import MulticoilForwardMRI, get_mvue
from utils import ifft, normalize, normalize_np, unnormalize

from tqdm import tqdm
from aux_motion_data import get_motion_data, TruncatedMVUE
from matplotlib import pyplot as plt

nmse_loss = lambda x, y: np.sum(np.square(np.abs(x-y)), axis=(-1, -2)) /\
    np.sum(np.square(np.abs(x)), axis=(-1, -2))
from skimage.metrics import structural_similarity as ssim_loss
from skimage.metrics import peak_signal_noise_ratio as psnr_loss

# Seeds
torch.manual_seed(2021)
np.random.seed(2021)

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1) # !!! Must be '1' for knee
parser.add_argument('--batches', nargs='+', type=int, default=[0, 0])
parser.add_argument('--extra_accel', type=int, default=1)
parser.add_argument('--accel_safety', type=int, default=0)
parser.add_argument('--noise_boost', type=float, default=0.1)
parser.add_argument('--normalize_grad', type=int, default=1)
parser.add_argument('--dc_boost', type=float, default=1.)
parser.add_argument('--step_lr', nargs='+', type=float, default=[9e-5])
parser.add_argument('--anatomy', type=str, default='knee')
parser.add_argument('--contrast', type=str, default='T2')
parser.add_argument('--val_num', type=int, default=0)
args   = parser.parse_args()

# Always !!!
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True
# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu);

# Load a diffusion model
target_file = './ncsnv2-mri-mvue/logs/mri-mvue/checkpoint_100000.pth'
# Model config
config        = DotMap()
config.device = 'cuda:0'
# Inner model
config.model.normalization = 'InstanceNorm++'
config.model.nonlinearity  = 'elu'
config.model.sigma_dist    = 'geometric'
config.model.sigma_begin   = 232
config.model.sigma_end     = 0.00066
config.model.num_classes   = 2311
config.model.ngf           = 128
# Data
meta_step_lr = args.step_lr
config.sampling_batch_size     = args.batch_size # 12 fills 10.8 GB
config.sampling.log_steps      = 5
config.sampling.snapshot_steps = 100
# Inference
config.inference.sigma_offset   = 800 # !!! Skip some early (huge) sigmas
config.inference.num_steps_each = 4   # !!! More budget here
config.inference.noise_boost    = args.noise_boost
config.inference.num_steps      = \
    config.model.num_classes - config.inference.sigma_offset # Leftover

# Data
config.data.channels = 2
# Model
diffuser = NCSNv2Deepest(config)
diffuser = torch.nn.DataParallel(diffuser)
# Load weights
model_state = torch.load(target_file)
diffuser.load_state_dict(model_state[0], strict=True)
# Switch to eval mode and extract module
diffuser = diffuser.module
diffuser.eval()
# Dummy dataset used to get knee masks
val_dataset = TruncatedMVUE([], [], [], [])


global_ksp, global_maps, global_gt, global_slices, global_filenames, global_motion = \
    get_motion_data(args.anatomy, args.contrast, args.val_num)


# Target specific batches
target_batches = np.arange(args.batches[0], args.batches[1]+1)
# Number of actual extracted samples (across all batches)
num_real_samples = len(target_batches) * config.sampling_batch_size

# For each hyperparameter
for idx, local_step_lr in enumerate(meta_step_lr):
    # Set configuration
    config.sampling.step_lr = local_step_lr

    # Global metrics
    num_metric_steps     = int(np.ceil((config.inference.num_steps) /\
                                        config.sampling.log_steps))
    oracle_unmasked_nmse = np.zeros((num_real_samples, num_metric_steps))
    oracle_masked_nmse   = np.zeros((num_real_samples, num_metric_steps))
    oracle_unmasked_ssim = np.zeros((num_real_samples, num_metric_steps))
    oracle_masked_ssim   = np.zeros((num_real_samples, num_metric_steps))
    oracle_unmasked_psnr = np.zeros((num_real_samples, num_metric_steps))
    oracle_masked_psnr   = np.zeros((num_real_samples, num_metric_steps))
    # Data consistency log
    dc_log = np.zeros((num_real_samples, config.inference.num_steps))

    # Global outputs
    num_log_steps        = int(np.ceil((config.inference.num_steps) /\
                                        config.sampling.snapshot_steps))
    if args.anatomy == 'brain':
        logged_out           = np.zeros((
            num_real_samples, num_log_steps,
            global_ksp.shape[-2], global_ksp.shape[-1]), dtype=np.complex64)
        our_gt              = np.zeros((num_real_samples,
            global_ksp.shape[-2], global_ksp.shape[-1]), dtype=np.complex64)
        our_motion              = np.zeros((num_real_samples,
            global_ksp.shape[-2], global_ksp.shape[-1]), dtype=np.complex64)
        print("our_gt shape:", our_gt.shape)
    elif args.anatomy == 'knee':
        assert args.batch_size == 1, 'For knee, batch size has to be 1!'
        # Universal crop size
        crop_size = 320
        logged_out           = np.zeros((
            num_real_samples, num_log_steps,
            crop_size, crop_size), dtype=np.complex64)
        our_gt              = np.zeros((num_real_samples,
            crop_size, crop_size), dtype=np.complex64)

    # Results
    result_dir = 'full_motion_results_Dec8/val_num_%d/grid_search/normMeas%d_%s_extraAccel%d_dcBoost%.1f' % (
        args.val_num, args.normalize_grad, args.anatomy, args.extra_accel, args.dc_boost)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # For each batch
    for rolling_idx, batch_idx in enumerate(target_batches):
        # Sample range
        sample_range = np.arange(batch_idx * config.sampling_batch_size,
             np.minimum((batch_idx + 1) * config.sampling_batch_size,
                        len(global_ksp)))
        if args.anatomy == 'knee':
            sample_range = int(sample_range[0])
            log_range    = np.asarray([rolling_idx])
        else:
            # Special skip
            if len(sample_range) == 0:
                print('Skipping an invalid (too large idx) batch! Watch your args!!')
                continue
            log_range = np.arange(rolling_idx * config.sampling_batch_size,
                 rolling_idx * config.sampling_batch_size + len(sample_range))

        # Batch and undersample further
        local_ksp  = torch.tensor(global_ksp[sample_range],
                                  dtype=torch.complex64).cuda()
        local_maps = torch.tensor(global_maps[sample_range],
                                  dtype=torch.complex64).cuda()
        local_gt   = global_gt[sample_range]
        local_motion = global_motion[sample_range]
        # Fake axis for knee
        if args.anatomy == 'knee':
            local_ksp, local_maps, local_gt = \
                local_ksp[None, :], local_maps[None, :], local_gt[None, :]

        # Retrospective acceleration
        if args.extra_accel > 1.01: # Safety
            if args.anatomy == 'brain':
                # Find non-zero indices
                for sample_idx in range(local_ksp.shape[0]):
                    center_lines = np.arange(
                        int(local_ksp.shape[-1]/2 - 0.04 * local_ksp.shape[-1]),
                        int(local_ksp.shape[-1]/2 + 0.04 * local_ksp.shape[-1]))
                    nonzero_idx = np.where(
                        np.sum(np.abs(local_ksp[sample_idx].cpu().numpy()),
                               axis=(0, 1)) > 1e-30)[0]
                    # !!! Don't touch the center lines
                    if bool(args.accel_safety):
                        nonzero_idx = np.setdiff1d(nonzero_idx, center_lines)
                    # Downsample
                    local_ksp[sample_idx, :, :, nonzero_idx[::args.extra_accel]] = 0.

            elif args.anatomy == 'knee':
                # How many lines reserved as ACS
                acs_size = 0.08 if args.extra_accel < 8 else 0.04
                # Get and apply mask
                central_lines = int(np.floor(local_ksp.shape[-1] * acs_size))
                mask          = torch.tensor(val_dataset._get_mask(
                    acs_lines=central_lines, total_lines=int(local_ksp.shape[-1]),
                    R=args.extra_accel, pattern='random')).type(torch.float32).cuda()

                # Mask k-space and estimate maps
                local_ksp = local_ksp * mask[None, None, None, ...]
                # Fill in mask to the right sizes
                masks = mask[None, ...]
                # Optional eval masks
                local_eval_masks = \
                    np.abs(local_gt) > np.percentile(np.abs(local_gt), 5)

        if args.anatomy == 'brain':
            # Get masks for evaluation - curently based on exact zeroes
            local_eval_masks = np.abs(local_gt) > 1e-30
            # Extract sampling masks
            masks = (torch.sum(
                torch.abs(local_ksp), dim=(1, 2)) > 1e-30).type(torch.float32)

        # Store GT
        print('local_gt shape:', local_gt.shape)
        our_gt[log_range] = np.copy(local_gt)
        our_motion[log_range] = np.copy(local_motion)
        # Get ZF MVUE
        with torch.no_grad():
            estimated_mvue = torch.tensor(get_mvue(local_ksp.cpu().numpy(),
                        local_maps.cpu().numpy())).cuda()

        # Get a forward operator
        forward_operator = lambda x: MulticoilForwardMRI('vertical')(
                torch.complex(x[:, 0], x[:, 1]), local_maps, masks)

        # Initial samples
        samples = torch.rand(local_ksp.shape[0], config.data.channels,
                             local_ksp.shape[-2], local_ksp.shape[-1],
                             dtype=torch.float32).cuda()
        # Rolling indices
        log_idx, snap_idx = 0, 0

        # Inference
        with torch.no_grad():
            # For each noise level
            for noise_idx in tqdm(range(config.inference.num_steps)):
                # Get current noise power
                sigma  = diffuser.sigmas[noise_idx + config.inference.sigma_offset]
                labels = torch.ones(samples.shape[0],
                                    device=samples.device) * \
                    (noise_idx + config.inference.sigma_offset)
                labels = labels.long()
                step_size = config.sampling.step_lr * (
                    sigma / diffuser.sigmas[-1]) ** 2
                # For each step spent there
                for step_idx in range(config.inference.num_steps_each):
                    # Generate noise
                    noise = torch.randn_like(samples) * \
                        torch.sqrt(args.noise_boost * step_size * 2)
                    # get score from model
                    p_grad = diffuser(samples, labels)

                    # get measurements and DC loss for current estimate
                    meas    = forward_operator(normalize(samples, estimated_mvue))
                    dc_loss = meas - local_ksp

                    # Two different ways
                    if bool(args.normalize_grad):
                        # compute gradient, i.e., gradient = A_adjoint * ( y - Ax_hat )
                        meas_grad = 2 * torch.view_as_real(
                            torch.sum(ifft(dc_loss) * \
                                      torch.conj(local_maps), axis=1)
                                ).permute(0, 3, 1, 2)
                        # Normalize
                        meas_grad = meas_grad / torch.norm(meas_grad)
                        meas_grad = meas_grad * torch.norm(p_grad)
                    else:
                        # compute gradient, i.e., gradient = A_adjoint * ( y - Ax_hat )
                        meas_grad = 2 * torch.view_as_real(
                            torch.sum(ifft(dc_loss) * \
                                      torch.conj(local_maps), axis=1) / \
                                (sigma ** 2)).permute(0, 3, 1, 2)
                        # re-normalize, since measuremenets are from a normalized estimate
                        meas_grad = unnormalize(meas_grad, estimated_mvue)

                    # combine measurement gradient, prior gradient and noise
                    samples = samples + step_size * (
                        p_grad - args.dc_boost * meas_grad) + noise

                # Always log DC
                dc_log[log_range, noise_idx] = \
                    torch.sum(torch.square(
                        torch.abs(dc_loss)), dim=(1, 2, 3)).cpu().numpy()
                # Get output and log periodically
                if np.mod(noise_idx, config.sampling.log_steps) == 0:
                    # Don't forget to normalize!
                    cplx_out = normalize_np(
                        (samples[:, 0] + 1j * samples[:, 1]).cpu().numpy(),
                        local_gt)
                    # If knee, crop
                    if args.anatomy == 'knee':
                        cplx_out = sp.resize(cplx_out,
                             (cplx_out.shape[0], crop_size, crop_size))

                    # Log metrics sample-by-sample
                    for idx, loc_idx in enumerate(log_range):
                        # Log semi-masked metrics
                        oracle_unmasked_nmse[loc_idx, log_idx] = \
                            nmse_loss(np.abs(local_gt[idx]), np.abs(cplx_out[idx]))
                        oracle_unmasked_ssim[loc_idx, log_idx] = \
                            ssim_loss(np.abs(local_gt[idx]), np.abs(cplx_out[idx]),
                                 data_range=np.abs(local_gt[idx]).max())
                        oracle_unmasked_psnr[loc_idx, log_idx] = \
                            psnr_loss(np.abs(local_gt[idx]), np.abs(cplx_out[idx]),
                                 data_range=np.abs(local_gt[idx]).max())

                        # Log semi-masked metrics
                        oracle_masked_nmse[loc_idx, log_idx] = \
                            nmse_loss(np.abs(local_gt[idx]) * local_eval_masks[idx],
                                      np.abs(cplx_out[idx]) * local_eval_masks[idx])
                        oracle_masked_ssim[loc_idx, log_idx] = \
                            ssim_loss(np.abs(local_gt[idx]) * local_eval_masks[idx],
                                      np.abs(cplx_out[idx]) * local_eval_masks[idx],
                                 data_range=np.abs(local_gt[idx]).max())
                        oracle_masked_psnr[loc_idx, log_idx] = \
                            psnr_loss(np.abs(local_gt[idx]) * local_eval_masks[idx],
                                      np.abs(cplx_out[idx]) * local_eval_masks[idx],
                                 data_range=np.abs(local_gt[idx]).max())
                    log_idx = log_idx + 1

                    # Only log raw output rarer
                    if np.mod(noise_idx, config.sampling.snapshot_steps) == 0:
                        logged_out[log_range, snap_idx] = np.copy(cplx_out)
                        snap_idx = snap_idx + 1

    # Save to file
    filename = result_dir + '/R3_startBatch%d_endBatch%d_noise%.2e_step%.2e.pt' % (
        args.batches[0], args.batches[-1], args.noise_boost, local_step_lr)
    torch.save({'oracle_unmasked_nmse': oracle_unmasked_nmse,
                'oracle_unmasked_ssim': oracle_unmasked_ssim,
                'oracle_unmasked_psnr': oracle_unmasked_psnr,
                'oracle_masked_nmse': oracle_masked_nmse,
                'oracle_masked_ssim': oracle_masked_ssim,
                'oracle_masked_psnr': oracle_masked_psnr,
                'dc_log': dc_log,
                'logged_out': logged_out,
                'global_slices': global_slices,
                'global_filenames': global_filenames,
                'global_gt': our_gt,
                'global_motion': our_motion,
                'args': args,
                'config': config}, filename)
