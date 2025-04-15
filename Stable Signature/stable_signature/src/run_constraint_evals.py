import argparse
import os
import torch
from finetune_ldm_decoder import main

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./../data/COCO_finetune_train_3k/',
                      help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, default='./../data/COCO_finetune_val_1k/',
                      help='Path to validation data directory')
    parser.add_argument('--ldm_config', type=str, default='./../models/sd_config.yaml',
                      help='Path to LDM config file')
    parser.add_argument('--ldm_ckpt', type=str, default='./../models/sd_checkpoint.ckpt',
                      help='Path to LDM checkpoint')
    parser.add_argument('--msg_decoder_path', type=str, default='./../models/decoder.pth',
                      help='Path to message decoder')
    parser.add_argument('--loss_formation', type=str, default='constrained',
                      choices=['regular', 'constrained'],
                      help='Loss formation type')
    parser.add_argument('--image_loss_constraint', type=float, nargs='+', required=True,
                      help='One or more image loss constraint thresholds')
    parser.add_argument('--dual_lr', type=float, default=0.02,
                      help='Dual learning rate')
    parser.add_argument('--steps', type=int, default=1000,
                      help='Number of training steps')
    parser.add_argument('--cuda', type=int, default=0,
                      help='CUDA device ID')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    # Run main function for each constraint value
    for constraint in args.image_loss_constraint:
        print(f"\nRunning evaluation with image loss constraint: {constraint}")
        # Create a copy of args with the current constraint value
        current_args = argparse.Namespace(**vars(args))
        current_args.image_loss_constraint = constraint
        main(current_args)
