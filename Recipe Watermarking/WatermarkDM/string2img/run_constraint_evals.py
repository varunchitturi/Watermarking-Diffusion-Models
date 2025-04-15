import argparse
import os
import torch
from train import main

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1,
                      help='CUDA device ID')
    parser.add_argument('--use_pretrained', type=str, default='wm',
                      help='Pretrained model type')
    parser.add_argument('--data_dir', type=str, default='../edm/datasets/uncompressed/COCO',
                      help='Path to data directory')
    parser.add_argument('--image_resolution', type=int, default=256,
                      help='Image resolution')
    parser.add_argument('--output_dir', type=str, default='./_output/COCO',
                      help='Output directory')
    parser.add_argument('--loss_formation', type=str, default='constrained',
                      choices=['regular', 'constrained'],
                      help='Loss formation type')
    parser.add_argument('--image_loss_constraint', type=float, nargs='+', required=True,
                      help='One or more image loss constraint thresholds')
    parser.add_argument('--dual_lr', type=float, default=0.07,
                      help='Dual learning rate')
    parser.add_argument('--primal_per_dual', type=int, default=10,
                      help='Number of primal steps per dual step')
    parser.add_argument('--image_loss', type=str, default='watson-vgg',
                      help='Image loss type')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=70,
                      help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                      help='Learning rate')
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
