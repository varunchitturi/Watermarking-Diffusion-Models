# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import sys
from copy import deepcopy
from omegaconf import OmegaConf
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional
from augly.image import functional as aug_functional
from torchvision.utils import save_image
import collections
import stegastamp_models

import utils
import utils_img
import utils_model

sys.path.append('src')
from ldm.models.autoencoder import AutoencoderKL
from ldm.models.diffusion.ddpm import LatentDiffusion
from loss.loss_provider import LossProvider

import wandb

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Data parameters')
    aa("--train_dir", type=str, help="Path to the training data directory", required=True)
    aa("--val_dir", type=str, help="Path to the validation data directory", required=True)

    group = parser.add_argument_group('Model parameters')
    aa("--ldm_config", type=str, default="sd/stable-diffusion-v-1-4-original/v1-inference.yaml", help="Path to the configuration file for the LDM model") 
    aa("--ldm_ckpt", type=str, default="sd/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt", help="Path to the checkpoint file for the LDM model") 
    aa("--msg_decoder_path", type=str, default= "models/hidden/dec_48b_whit.torchscript.pt", help="Path to the hidden decoder for the watermarking model")
    aa("--decoder_model_type", type=str, choices=["hidden", "stegastamp"], default="hidden", help="Type of watermarking model")
    aa("--num_bits", type=int, default=48, help="Number of bits in the watermark")
    aa("--redundancy", type=int, default=1, help="Number of times the watermark is repeated to increase robustness")
    aa("--decoder_depth", type=int, default=8, help="Depth of the decoder in the watermarking model")
    aa("--decoder_channels", type=int, default=64, help="Number of channels in the decoder of the watermarking model")

    group = parser.add_argument_group('Training parameters')
    aa("--batch_size", type=int, default=2, help="Batch size for training")
    aa("--img_size", type=int, default=256, help="Resize images to this size")
    aa("--loss_i", type=str, default="watson-vgg", help="Type of loss for the image loss. Can be watson-vgg, l2, watson-dft, etc.")
    aa("--loss_w", type=str, default="bce", help="Type of loss for the watermark loss. Can be mse or bce")
    
    aa("--lambda_w", type=float, default=1.0, help="Weight of the watermark loss in the total loss")
    aa("--optimizer", type=str, default="AdamW,lr=1e-5", help="Optimizer and learning rate for training")
    aa("--steps", type=int, default=100, help="Number of steps to train the model for")
    aa("--warmup_steps", type=int, default=20, help="Number of warmup steps for the optimizer")
    
    aa("--loss_formation", type=str, required=True, choices=["regular", "constrained"], help="Whether to use constrained learning or a single objective with a regularization term.")
    # regular loss formation
    aa("--image_loss_weight", type=float, help="Weight of the image loss in the total loss")
    # constrained loss formation
    aa("--image_loss_constraint", type=float, help="The threshold for the image loss constraint")
    aa("--dual_lr", type=float, help="The dual learning rate when using constrained learning.")
    aa("--primal_per_dual", type=int, default=5, help="The number of primal learning steps per dual learning step")

    aa("--transform", type=str, default="none", choices=["none", "random"], help="Type of transform to apply to the images")
    
    group = parser.add_argument_group('Logging and saving freq. parameters')
    aa("--log_freq", type=int, default=10, help="Logging frequency (in steps)")
    aa("--save_img_freq", type=int, default=1000, help="Frequency of saving generated images (in steps)")

    group = parser.add_argument_group('Experiments parameters')
    aa("--num_keys", type=int, default=1, help="Number of fine-tuned checkpoints to generate")
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    aa("--seed", type=int, default=0)
    aa("--debug", type=utils.bool_inst, default=False, help="Debug mode")

    return parser


def main(params):
    
    run = wandb.init(
        project="Constrained Stable Signature",
        config=params,
        reinit=True
    )

    # Set seeds for reproductibility 
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)
    
    # check if arguments are valid
    if params.loss_formation == "regular" and (params.image_loss_weight is None):
        raise Exception("params.image_loss_weight is None")
    elif params.loss_formation == "constrained" and (params.image_loss_constraint is None or params.dual_lr is None or params.primal_per_dual is None):
        raise Exception("One or more of params.image_loss_constraint, params.dual_lr, or params.primal_per_dual is None")
    
    # Print the arguments
    print("__git__:{}".format(utils.get_sha()))
    print("__log__:{}".format(json.dumps(vars(params))))

    # Create the directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    imgs_dir = os.path.join(params.output_dir, 'imgs')
    params.imgs_dir = imgs_dir
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir, exist_ok=True)

    # Loads LDM auto-encoder models
    print(f'>>> Building LDM model with config {params.ldm_config} and weights from {params.ldm_ckpt}...')
    config = OmegaConf.load(f"{params.ldm_config}")
    ldm_ae: LatentDiffusion = utils_model.load_model_from_config(config, params.ldm_ckpt)
    ldm_ae: AutoencoderKL = ldm_ae.first_stage_model
    ldm_ae.eval()
    ldm_ae.to(device)
    
    # Loads hidden decoder
    print(f'>>> Building hidden decoder with weights from {params.msg_decoder_path}...')
    if 'torchscript' in params.msg_decoder_path: 
        msg_decoder = torch.jit.load(params.msg_decoder_path).to(device)
        # already whitened
        
    elif params.decoder_model_type == "hidden":
        msg_decoder = utils_model.get_hidden_decoder(num_bits=params.num_bits, redundancy=params.redundancy, num_blocks=params.decoder_depth, channels=params.decoder_channels).to(device)
        ckpt = utils_model.get_hidden_decoder_ckpt(params.msg_decoder_path)
        print(msg_decoder.load_state_dict(ckpt, strict=False))
        msg_decoder.eval()

        # whitening
        print(f'>>> Whitening...')
        with torch.no_grad():
            # features from the dataset
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            loader = utils.get_dataloader(params.train_dir, transform, batch_size=16, collate_fn=None)
            ys = []
            for i, x in enumerate(loader):
                x = x.to(device)
                y = msg_decoder(x)
                ys.append(y.to('cpu'))
                
            ys = torch.cat(ys, dim=0)
            nbit = ys.shape[1]
            
            # whitening
            mean = ys.mean(dim=0, keepdim=True) # NxD -> 1xD
            ys_centered = ys - mean # NxD
            cov = ys_centered.T @ ys_centered
            e, v = torch.linalg.eigh(cov)
            L = torch.diag(1.0 / torch.pow(e, exponent=0.5))
            weight = torch.mm(L, v.T)
            bias = -torch.mm(mean, weight.T).squeeze(0)
            linear = nn.Linear(nbit, nbit, bias=True)
            linear.weight.data = np.sqrt(nbit) * weight
            linear.bias.data = np.sqrt(nbit) * bias
            msg_decoder = nn.Sequential(msg_decoder, linear.to(device))
            torchscript_m = torch.jit.script(msg_decoder)
            params.msg_decoder_path = params.msg_decoder_path.replace(".pth", "_whit.pth")
            print(f'>>> Creating torchscript at {params.msg_decoder_path}...')
            torch.jit.save(torchscript_m, params.msg_decoder_path)
    elif params.decoder_model_type == "stegastamp":
        msg_decoder = stegastamp_models.StegaStampDecoder(
        params.img_size,
        3,
        params.num_bits,
        )
        msg_decoder_load = torch.load(params.msg_decoder_path)
        if type(msg_decoder_load) is collections.OrderedDict:
            msg_decoder.load_state_dict(msg_decoder_load)
        else:
            msg_decoder = msg_decoder_load
        msg_decoder.to(device)
    
    msg_decoder.eval()
    nbit = msg_decoder(torch.zeros(1, 3, params.img_size, params.img_size).to(device)).shape[-1]

    # Freeze LDM and hidden decoder
    for param in [*msg_decoder.parameters(), *ldm_ae.parameters()]:
        param.requires_grad = False

    # Loads the data
    print(f'>>> Loading data from {params.train_dir} and {params.val_dir}...')
    vqgan_transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
        utils_img.normalize_vqgan,
    ])
    train_loader = utils.get_dataloader(params.train_dir, vqgan_transform, params.batch_size, num_imgs=params.batch_size*params.steps, shuffle=True, num_workers=4, collate_fn=None)
    val_loader = utils.get_dataloader(params.val_dir, vqgan_transform, params.batch_size*4, num_imgs=1000, shuffle=False, num_workers=4, collate_fn=None)
    vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])
    
    # Create losses
    print(f'>>> Creating losses...')
    print(f'Losses: {params.loss_w} and {params.loss_i}...')
    if params.loss_w == 'mse':        
        loss_w = lambda decoded, keys, temp=10.0: torch.mean((decoded*temp - (2*keys-1))**2) # b k - b k
    elif params.loss_w == 'bce':
        loss_w = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(decoded*temp, keys, reduction='mean')
    else:
        raise NotImplementedError
    
    if params.loss_i == 'l2':
        loss_i = lambda imgs_w, imgs: torch.mean((imgs_w - imgs)**2)
    elif params.loss_i == 'watson-dft':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    elif params.loss_i == 'watson-vgg':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    elif params.loss_i == 'ssim':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('SSIM', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    else:
        raise NotImplementedError

    for ii_key in range(params.num_keys):

        # Creating key
        print(f'\n>>> Creating key with {nbit} bits...')
        key = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device=device)
        key_str = "".join([ str(int(ii)) for ii in key.tolist()[0]])
        print(f'Key: {key_str}')

        # Copy the LDM decoder and finetune the copy
        ldm_decoder = deepcopy(ldm_ae)
        ldm_decoder.encoder = nn.Identity()
        ldm_decoder.quant_conv = nn.Identity()
        ldm_decoder.to(device)
        for param in ldm_decoder.parameters():
            param.requires_grad = True
        optim_params = utils.parse_params(params.optimizer)
        optimizer = utils.build_optimizer(model_params=ldm_decoder.parameters(), **optim_params)

        # Training loop
        print(f'>>> Training...')
                
        train_stats = train(train_loader, optimizer, loss_w, loss_i, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, params)
        val_stats = val(val_loader, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, params)
        log_stats = {'key': key_str,
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in val_stats.items()},
            }
        save_dict = {
            'ldm_decoder': ldm_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
        }

        # Save checkpoint
        torch.save(save_dict, os.path.join(params.output_dir, f"checkpoint_{ii_key:03d}.pth"))
        with (Path(params.output_dir) / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
        with (Path(params.output_dir) / "keys.txt").open("a") as f:
            f.write(os.path.join(params.output_dir, f"checkpoint_{ii_key:03d}.pth") + "\t" + key_str + "\n")
        print('\n')
    
    run.finish()
    
def random_transform(imgs):
    def sample_params(self, x):
        # randomly select one of augmentations
        ps = np.array([1,1,1,1,1])
        ps = ps / ps.sum()
        augm_type = np.random.choice(['none', 'rotation', 'crop', 'resize', 'blur'], p=ps)
        # flip param
        f = np.random.rand()>0.5 if self.flip else 0  
        # sample params
        if augm_type == 'none':
            return augm_type, 0, f
        elif augm_type == 'rotation':
            d = np.random.vonmises(0, 1)*self.degrees/np.pi
            return augm_type, d, f
        elif augm_type in ['crop', 'resize']:
            width, height = functional.get_image_size(x)
            area = height * width
            target_area = np.random.uniform(*self.crop_scale) * area
            aspect_ratio = np.exp(np.random.uniform(np.log(self.crop_ratio[0]), np.log(self.crop_ratio[1])))
            tw = int(np.round(np.sqrt(target_area * aspect_ratio)))
            th = int(np.round(np.sqrt(target_area / aspect_ratio)))
            if augm_type == 'crop':
                i = np.random.randint(0, max(min(height - th + 1, height-1), 0)+1)
                j = np.random.randint(0, max(min(width - tw + 1, width-1), 0)+1)
                return augm_type, (i ,j, th, tw), f
            elif augm_type == 'resize':
                s = np.random.uniform(*self.resize_scale)
                return augm_type, (s, th, tw), f
        elif augm_type == 'blur':
            b = np.random.randint(1, self.blur_size+1)
            b = b-(1-b%2) # make it odd         
            return augm_type, b, f
        
    def apply(self, x, augmentation):
        augm_type, param, f = augmentation
        if augm_type == 'blur':
            x = functional.gaussian_blur(x, param)
        if augm_type == 'rotation':
            x = functional.rotate(x, param, interpolation=self.interpolation)
            # x = functional.rotate(x, d, expand=True, interpolation=self.interpolation)
        elif augm_type == 'crop':
            x = functional.crop(x, *param)
        elif augm_type == 'resize':
            s, h, w = param
            x = functional.resize(x, int((s**0.5)*min(h,w)), interpolation=self.interpolation)
        x = functional.hflip(x) if f else x
        return x
    
    transform_params = [sample_params(x) for x in imgs]
    imgs_aug = [apply(x, param) for x, param in zip(imgs, transform_params)]
    return imgs_aug
        

    

    

def train(data_loader: Iterable, optimizer: torch.optim.Optimizer, loss_w: Callable, loss_i: Callable, ldm_ae: AutoencoderKL, ldm_decoder:AutoencoderKL, msg_decoder: nn.Module, vqgan_to_imnet:nn.Module, key: torch.Tensor, params: argparse.Namespace):
    header = 'Train'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.train()
    base_lr = optimizer.param_groups[0]["lr"]
    
    if params.loss_formation == "regular":
        primal_steps = 1
        image_loss_weight = params.image_loss_weight
    else:
        primal_steps = params.primal_per_dual
        image_loss_weight = 0

    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        
        utils.adjust_learning_rate(optimizer, ii, params.steps, params.warmup_steps, base_lr)
        
        
        for _ in range(primal_steps):
            
            imgs = imgs.to(device)
            keys = key.repeat(imgs.shape[0], 1)
            # encode images
            imgs_z = ldm_ae.encode(imgs) # b c h w -> b z h/f w/f
            imgs_z = imgs_z.mode()

            # decode latents with original and finetuned decoder
            imgs_d0 = ldm_ae.decode(imgs_z) # b z h/f w/f -> b c h w
            
            imgs_w = ldm_decoder.decode(imgs_z) # b z h/f w/f -> b c h w
            

            # extract watermark
            if params.transform == "random":
                decoded = msg_decoder(random_transform(vqgan_to_imnet(imgs_w))) # b c h w -> b k
            elif params.transform == "none":
                decoded = msg_decoder(vqgan_to_imnet(imgs_w)) # b c h w -> b k
            
        
            # compute loss
            lossw = loss_w(decoded, keys)
            lossi = loss_i(imgs_w, imgs_d0)
            
            loss = params.lambda_w * lossw + image_loss_weight * lossi

            # optim step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        if params.loss_formation == "constrained":
            image_loss_weight = max(0, image_loss_weight + 
                                    params.dual_lr * (lossi.detach().item() - params.image_loss_constraint))

        # log stats
        diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        word_accs = (bit_accs == 1) # b
        log_stats = {
            "loss": loss.item(),
            "signature_loss": lossw.item(),
            "image_loss": lossi.item(),
            "image_loss_weight": image_loss_weight,
            "psnr": utils_img.psnr(imgs_w, imgs_d0).mean().item(),
            # "psnr_ori": utils_img.psnr(imgs_w, imgs).mean().item(),
            "signature_acc_avg": torch.mean(bit_accs).item(),
            "word_acc_avg": torch.mean(word_accs.type(torch.float)).item(),
            "lr": optimizer.param_groups[0]["lr"]
        }
        
        wandb_stats = log_stats.copy()
        wandb_stats["original_images"] = wandb.Image(imgs, caption=f"Original Images")
        wandb_stats["decoded_original_images"] = wandb.Image(imgs_d0, caption=f"Decoded Images by Original Model")
        wandb_stats["decoded_watermarked_images"] = wandb.Image(imgs_w, caption=f"Decoded Images by Finetuned Model (Watermarked)")
        wandb.log(wandb_stats)
        for name, loss in log_stats.items():
            if "images" not in name:
                metric_logger.update(**{name:loss})
        if ii % params.log_freq == 0:
            print(json.dumps(log_stats))
        
        # save images during training
        if ii % params.save_img_freq == 0:
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs),0,1), os.path.join(params.imgs_dir, f'{ii:03}_train_orig.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0),0,1), os.path.join(params.imgs_dir, f'{ii:03}_train_d0.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w),0,1), os.path.join(params.imgs_dir, f'{ii:03}_train_w.png'), nrow=8)
    
    print("Averaged {} stats:".format('train'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def val(data_loader: Iterable, ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL, msg_decoder: nn.Module, vqgan_to_imnet:nn.Module, key: torch.Tensor, params: argparse.Namespace):
    header = 'Eval'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.eval()
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        
        imgs = imgs.to(device)

        imgs_z = ldm_ae.encode(imgs) # b c h w -> b z h/f w/f
        imgs_z = imgs_z.mode()

        imgs_d0 = ldm_ae.decode(imgs_z) # b z h/f w/f -> b c h w
        imgs_w = ldm_decoder.decode(imgs_z) # b z h/f w/f -> b c h w
        
        keys = key.repeat(imgs.shape[0], 1)

        log_stats = {
            "iteration": ii,
            "psnr": utils_img.psnr(imgs_w, imgs_d0).mean().item(),
            # "psnr_ori": utils_img.psnr(imgs_w, imgs).mean().item(),
        }
        attacks = {
            'none': lambda x: x,
            'crop_01': lambda x: utils_img.center_crop(x, 0.1),
            'crop_05': lambda x: utils_img.center_crop(x, 0.5),
            'rot_15': lambda x: utils_img.rotate(x, 15),
            'rot_30': lambda x: utils_img.rotate(x, 30),
            'rot_45': lambda x: utils_img.rotate(x, 45),
            'rot_60': lambda x: utils_img.rotate(x, 60),
            'rot_75': lambda x: utils_img.rotate(x, 75),
            'rot_90': lambda x: utils_img.rotate(x, 90),
            'resize_03': lambda x: utils_img.resize(x, 0.3),
            'resize_07': lambda x: utils_img.resize(x, 0.7),
            'blur_1': lambda x: functional.gaussian_blur(x, 1),
            'blur_3': lambda x: functional.gaussian_blur(x, 3),
            'blur_5': lambda x: functional.gaussian_blur(x, 5),
            'blur_7': lambda x: functional.gaussian_blur(x, 7),
            'brightness_p5': lambda x: utils_img.adjust_brightness(x, 0.5),
            'brightness_1p5': lambda x: utils_img.adjust_brightness(x, 1.5),
            'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
            'contrast_05': lambda x: utils_img.adjust_contrast(x, 0.5),
            'contrast_15': lambda x: utils_img.adjust_contrast(x, 1.5),
            'contrast_2': lambda x: utils_img.adjust_contrast(x, 2),
            'jpeg_10': lambda x: utils_img.jpeg_compress(x, 10),
            'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
            'jpeg_80': lambda x: utils_img.jpeg_compress(x, 80),
            'hue_p05': lambda x: utils_img.adjust_hue(x, -0.5),
            'hue_p025': lambda x: utils_img.adjust_hue(x, -0.25),
            'hue_0': lambda x: utils_img.adjust_hue(x, 0),
            'hue_025': lambda x: utils_img.adjust_hue(x, 0.25),
            'hue_05': lambda x: utils_img.adjust_hue(x, 0.5),
            'hue_1': lambda x: utils_img.adjust_hue(x, 1)
        }
        for name, attack in attacks.items():
            if params.decoder_model_type == "hidden" or name not in ['crop_01', 'crop_05', 'resize_03', 'resize_07']:
                imgs_aug = attack(vqgan_to_imnet(imgs_w))
                decoded = msg_decoder(imgs_aug) # b c h w -> b k
                diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
                bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
                word_accs = (bit_accs == 1) # b
                log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()
                log_stats[f'word_acc_{name}'] = torch.mean(word_accs.type(torch.float)).item()
        for name, loss in log_stats.items():
            metric_logger.update(**{name:loss})

        if ii % params.save_img_freq == 0:
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs),0,1), os.path.join(params.imgs_dir, f'{ii:03}_val_orig.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0),0,1), os.path.join(params.imgs_dir, f'{ii:03}_val_d0.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w),0,1), os.path.join(params.imgs_dir, f'{ii:03}_val_w.png'), nrow=8)
    
    print("Averaged {} stats:".format('eval'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
