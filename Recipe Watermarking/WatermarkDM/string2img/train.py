import argparse
import wandb
from loss.loss_provider import LossProvider


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, required=True, help="Directory with image dataset."
)
parser.add_argument(
    "--use_celeba_preprocessing",
    action="store_true",
    help="Use CelebA specific preprocessing when loading the images.",
)
parser.add_argument(
    "--output_dir", type=str, required=True, help="Directory to save results to."
)
parser.add_argument(
    "--bit_length",
    type=int,
    default=48,
    help="Number of bits in the fingerprint.",
)
parser.add_argument(
    "--image_resolution",
    type=int,
    default=32,
    help="Height and width of square images.",
)
parser.add_argument(
    "--num_epochs", type=int, default=10, help="Number of training epochs."
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
parser.add_argument("--cuda", type=str, default=0)


parser.add_argument(
    "--image_loss",
    type=str,
    default="l2",
    choices=["l2", "watson-vgg"],
    help="The criterion to use for the image loss"
)

parser.add_argument(
    "--loss_formation",
    type=str,
    required=True,
    choices=["regular", "constrained"],
    help="Whether to use constrained learning or a single objective with a regularization term. "
)

# Regular loss formation hyper parameters

parser.add_argument(
    "--image_loss_await",
    help="Train without image loss for the first x iterations",
    default=0,
    type=int
)

parser.add_argument(
    "--image_loss_weight",
    type=float,
    help="Image loss weight for image fidelity when using regular learning",
)

parser.add_argument(
    "--image_loss_ramp",
    type=int,
    default=3000,
    help="Linearly increase image loss weight over x iterations when using regular learning.",
)

# Constrained loss formation hyper parameters

parser.add_argument(
    "--image_loss_constraint",
    type=float,
    help="The threshold for the image loss constraint"
)

parser.add_argument(
    "--dual_lr",
    type=float,
    help="The dual learning rate when using constrained learning."
)

parser.add_argument(
    "--primal_per_dual",
    default=5,
    type=int,
    help="The number of primal learning steps per dual learning step"
)



args = parser.parse_args()


import glob
import os
from os.path import join
from time import time

from datetime import datetime

from tqdm import tqdm
import PIL

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from torch.optim import Adam

import models


LOGS_PATH = os.path.join(args.output_dir, "logs")
CHECKPOINTS_PATH = os.path.join(args.output_dir, "checkpoints")
SAVED_IMAGES = os.path.join(args.output_dir, "./saved_images")

writer = SummaryWriter(LOGS_PATH)

if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)
if not os.path.exists(CHECKPOINTS_PATH):
    os.makedirs(CHECKPOINTS_PATH)
if not os.path.exists(SAVED_IMAGES):
    os.makedirs(SAVED_IMAGES)


def generate_random_fingerprints(bit_length, batch_size=4, size=(400, 400)):
    z = torch.zeros((batch_size, bit_length), dtype=torch.float).random_(0, 2)
    return z


plot_points = (
    list(range(0, 1000, 100))
    + list(range(1000, 3000, 200))
    + list(range(3000, 100000, 1000))
)


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        if glob.glob(os.path.join(data_dir, "*.png")):
            self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        elif glob.glob(os.path.join(data_dir, "*.jpeg")):
            self.filenames = glob.glob(os.path.join(data_dir, "*.jpeg"))
        elif glob.glob(os.path.join(data_dir, "*.jpg")):
            self.filenames = glob.glob(os.path.join(data_dir, "*.jpg"))
        else:
            for i in range(50):
                if i == 0:
                    self.filenames = glob.glob(os.path.join(data_dir, f"{str(i).zfill(5)}", "*.png"))
                    self.filenames.extend(glob.glob(os.path.join(data_dir, f"{str(i).zfill(5)}", "*.jpeg")))
                    self.filenames.extend(glob.glob(os.path.join(data_dir, f"{str(i).zfill(5)}", "*.jpg")))
                else:
                    self.filenames.extend(glob.glob(os.path.join(data_dir, f"{str(i).zfill(5)}", "*.png")))
                    self.filenames.extend(glob.glob(os.path.join(data_dir, f"{str(i).zfill(5)}", "*.jpeg")))
                    self.filenames.extend(glob.glob(os.path.join(data_dir, f"{str(i).zfill(5)}", "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)


def load_data():
    global dataset, dataloader
    global IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, SECRET_SIZE

    IMAGE_RESOLUTION = args.image_resolution
    IMAGE_CHANNELS = 3

    SECRET_SIZE = args.bit_length

    if args.use_celeba_preprocessing:
        assert args.image_resolution == 128, f"CelebA preprocessing requires image resolution 128, got {args.image_resolution}."
        transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
    else:

        transform = transforms.Compose(
            [
                transforms.Resize(IMAGE_RESOLUTION),
                transforms.CenterCrop(IMAGE_RESOLUTION),
                transforms.ToTensor(),
            ]
        )

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")


def main():
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H:%M:%S")
    EXP_NAME = f"stegastamp_{args.bit_length}_{dt_string}"
    
    print("Training watermarking network - stegastamp with params:")
    print(args)
    
    wandb.init(
        project="Constrained Recipe Watermarking",
        config=args
    )

    device = torch.device("cuda")

    load_data()
    encoder = models.StegaStampEncoder(
        args.image_resolution,
        IMAGE_CHANNELS,
        args.bit_length,
        return_residual=False,
    )
    decoder = models.StegaStampDecoder(
        args.image_resolution,
        IMAGE_CHANNELS,
        args.bit_length,
    )
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    decoder_encoder_optim = Adam(
        params=list(decoder.parameters()) + list(encoder.parameters()), lr=args.lr
    )
    
    if args.loss_formation == "regular" and (args.image_loss_await is None or args.image_loss_weight is None or args.image_loss_ramp is None):
        raise Exception("One or more of args.image_loss_weight, args.image_loss_await, or args.image_loss_ramp is None")
    elif args.loss_formation == "constrained" and (args.image_loss_constraint is None or args.dual_lr is None or args.primal_per_dual is None):
        raise Exception("One or more of args.image_loss_constraint, args.dual_lr, or args.primal_per_dual is None")
    

    global_step = 0
    steps_since_image_loss_activated = -1

    image_loss_weight = 0
    if args.loss_formation == "regular":
        primal_steps = 1
    else:
        primal_steps = args.primal_per_dual
    

    for i_epoch in range(args.num_epochs):
        print(f"Epoch: {i_epoch}/{args.num_epochs}")
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
        for images, _ in tqdm(dataloader):
            
            global_step += 1
            
            batch_size = min(args.batch_size, images.size(0))
            
            fingerprints = generate_random_fingerprints(
                args.bit_length, batch_size, (args.image_resolution, args.image_resolution)
            )
            
            if args.loss_formation == "regular":
                image_loss_weight = min(
                    max(
                        0,
                        args.image_loss_weight
                        * (steps_since_image_loss_activated - args.image_loss_await)
                        / args.image_loss_ramp
                    ),
                    args.image_loss_weight,
                )
            elif args.loss_formation == "constrained" and steps_since_image_loss_activated > -1:
                image_loss_weight = max(0, 
                                        image_loss_weight
                                        + args.dual_lr
                                        * (image_loss.detach().item() - args.image_loss_constraint)
                                        )
                
            
            clean_images = images.to(device)
            fingerprints = fingerprints.to(device)

            for _ in range(primal_steps):
            
                fingerprinted_images = encoder(fingerprints, clean_images)

                decoder_output = decoder(fingerprinted_images)
                
                if args.image_loss == "l2":
                    criterion = nn.MSELoss()
                elif args.image_loss == "watson-vgg":
                    provider = LossProvider()
                    loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
                    loss_percep = loss_percep.to(device)
                    criterion = lambda imgs, fingerprinted_imgs: loss_percep((1+imgs)/2.0, (1+fingerprinted_imgs)/2.0)/ imgs.shape[0]
                    
                image_loss = criterion(fingerprinted_images, clean_images)

                criterion = nn.BCEWithLogitsLoss()
                BCE_loss = criterion(decoder_output.view(-1), fingerprints.view(-1))
                
                loss = BCE_loss + image_loss_weight * image_loss

                encoder.zero_grad()
                decoder.zero_grad()

                loss.backward()
                decoder_encoder_optim.step()
                
            fingerprints_predicted = (decoder_output > 0).float()
            bitwise_accuracy = 1.0 - torch.mean(
                torch.abs(fingerprints - fingerprints_predicted)
            )
            
            if steps_since_image_loss_activated == -1:
                if bitwise_accuracy.item() > 0.9:
                    steps_since_image_loss_activated = 0
            else:
                steps_since_image_loss_activated += 1

            wandb.log({
                "loss": loss.item(),
                "signature_loss": BCE_loss.item(),
                "signature_acc_avg": bitwise_accuracy.item(),
                "image_loss": image_loss.item(),
                "image_loss_weight": image_loss_weight,
                "steps_since_image_loss_activated": steps_since_image_loss_activated,
                "lr": decoder_encoder_optim.param_groups[0]["lr"],
                "original_images":  wandb.Image(clean_images, caption=f"Original Images"),
                "fingerprinted_images": wandb.Image(fingerprinted_images, caption=f"Fingerprinted Images")
            })
                
            if global_step in plot_points:
                save_image(
                    fingerprinted_images,
                    SAVED_IMAGES + "/{}.png".format(global_step),
                    normalize=True,
                )
            # checkpointing
            if global_step % 5000 == 0:
                torch.save(
                    decoder_encoder_optim.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_optim.pth"),
                )
                torch.save(
                    encoder.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_encoder.pth"),
                )
                torch.save(
                    decoder.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_decoder.pth"),
                )
                torch.save(
                    decoder.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_decoder.pth"),
                )
        
        print("Bitwise accuracy {}".format(bitwise_accuracy))
if __name__ == "__main__":
    main()
    