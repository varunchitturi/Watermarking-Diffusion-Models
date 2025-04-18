import argparse
import wandb
from loss.loss_provider import LossProvider
import collections


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
parser.add_argument(
    "--use_pretrained", default="", choices=["wm", "image", "wm_image"], help="Select a pretrained watermarking model to begin training with."
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

parser.add_argument(
    "--transform",
    type=str,
    default="none",
    choices=["none", "random"],
    help="Type of transform to apply to the images during training. 'none' means no transformation, 'random' applies random augmentations like rotation, crop, resize, blur, and flips."
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
from torchvision.transforms import functional
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
                transforms.ToTensor()
            ]
        )

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")
    
    
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

    device = torch.device(f"cuda:{args.cuda}")

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
    global_step = 0
    steps_since_image_loss_activated = -1
    if args.use_pretrained != "":
        steps_since_image_loss_activated = 0
        encoder_load = torch.load(f"pretrained_{args.use_pretrained}_stegastamp_encoder.pth")
        decoder_load = torch.load(f"pretrained_{args.use_pretrained}_stegastamp_decoder.pth")
        if type(encoder_load) is collections.OrderedDict:
            encoder.load_state_dict(encoder_load)
        else:
            encoder = encoder_load
        if type(decoder_load) is collections.OrderedDict:
            decoder.load_state_dict(decoder_load)
        else:
            decoder = decoder_load
            
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    decoder_encoder_optim = Adam(
        params=list(decoder.parameters()) + list(encoder.parameters()), lr=args.lr
    )
    
    if args.loss_formation == "regular" and (args.image_loss_await is None or args.image_loss_weight is None or args.image_loss_ramp is None):
        raise Exception("One or more of args.image_loss_weight, args.image_loss_await, or args.image_loss_ramp is None")
    elif args.loss_formation == "constrained" and (args.image_loss_constraint is None or args.dual_lr is None or args.primal_per_dual is None):
        raise Exception("One or more of args.image_loss_constraint, args.dual_lr, or args.primal_per_dual is None")
    

    image_loss_weight = 0
    if args.loss_formation == "regular":
        primal_steps = 1
    else:
        primal_steps = args.primal_per_dual
    
    if args.image_loss == "l2":
        image_criterion = nn.MSELoss()
    elif args.image_loss == "watson-vgg":
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        image_criterion = lambda imgs, fingerprinted_imgs: loss_percep((1+fingerprinted_imgs)/2.0, (1+imgs)/2.0)/ imgs.shape[0]
    signature_criterion = nn.BCEWithLogitsLoss()
    
    image_loss = torch.tensor(0)
    signature_loss = torch.tensor(0)
    loss = torch.tensor(0)
    
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

            for _ in range(1 if steps_since_image_loss_activated == -1 else primal_steps):
            
                fingerprinted_images = encoder(fingerprints, clean_images)
                
                if args.transform == "random":
                    fingerprinted_images = random_transform(fingerprinted_images)
                

                decoder_output = decoder(fingerprinted_images)
                
                    
                image_loss = image_criterion(fingerprinted_images, clean_images)

                
                signature_loss = signature_criterion(decoder_output.view(-1), fingerprints.view(-1))
                
                loss = signature_loss + image_loss_weight * image_loss

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
                "signature_loss": signature_loss.item(),
                "signature_acc_avg": bitwise_accuracy.item(),
                "image_loss": image_loss.item(),
                "image_loss_weight": image_loss_weight,
                "steps_since_image_loss_activated": steps_since_image_loss_activated,
                "lr": decoder_encoder_optim.param_groups[0]["lr"]
            })
                
            if global_step in plot_points:
                save_image(
                    fingerprinted_images,
                    SAVED_IMAGES + "/{}.png".format(global_step),
                    normalize=True,
                )
                wandb.log({
                    "original_images":  wandb.Image(clean_images, caption=f"Original Images"),
                    "fingerprinted_images": wandb.Image(fingerprinted_images, caption=f"Fingerprinted Images")
                })
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
    
