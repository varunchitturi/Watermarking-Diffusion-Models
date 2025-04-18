import argparse
import os
import torch
from main_multibit import main

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_formation', type=str, default='constrained',
                      choices=['regular', 'constrained'],
                      help='Loss formation type')
    parser.add_argument('--image_loss_constraint', type=float, nargs='+', required=True,
                      help='One or more image loss constraint thresholds')
    parser.add_argument('--dual_lr', type=float, default=0.2,
                      help='Dual learning rate')
    parser.add_argument('--image_loss', type=str, default='watson-vgg',
                      help='Image loss type')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size')
    parser.add_argument('--data_augmentation', type=str, default='none',
                      help='Data augmentation type')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    # Run main function for each constraint value
    for constraint in args.image_loss_constraint:
        print(f"\nRunning training with image loss constraint: {constraint}")
        # Create a copy of args with the current constraint value
        current_args = argparse.Namespace(**vars(args))
        current_args.image_loss_constraint = constraint
        main(current_args)
