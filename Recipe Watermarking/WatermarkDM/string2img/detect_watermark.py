import argparse
import glob
import PIL

parser = argparse.ArgumentParser()
parser.add_argument("--images_dir", type=str, required = True, help="Directory with images to decode fingerprints from.")
parser.add_argument("--note_file", type=str, required = True, help="File that contains mappings for the true fingerprints of each image in `--images_dir`.")

parser.add_argument(
    "--image_resolution",
    type=int,
    required = True,
    help="Height and width of square images.",
)
parser.add_argument(
    "--decoder_path",
    type=str,
    required=True,
    help="Path to trained StegaStamp decoder.",
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from time import time
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms


if int(args.cuda) == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda")


class CustomImageFolder(Dataset):
    def __init__(self, images_dir, note_file, transform=None):
        self.images_dir = images_dir
        self.note_file = note_file
        if glob.glob(os.path.join(images_dir, "*.png")):
            self.filenames = glob.glob(os.path.join(images_dir, "*.png"))
        elif glob.glob(os.path.join(images_dir, "*.jpeg")):
            self.filenames = glob.glob(os.path.join(images_dir, "*.jpeg"))
        elif glob.glob(os.path.join(images_dir, "*.jpg")):
            self.filenames = glob.glob(os.path.join(images_dir, "*.jpg"))
        else:
            for i in range(50):
                if i == 0:
                    self.filenames = glob.glob(os.path.join(images_dir, f"{str(i).zfill(5)}", "*.png"))
                    self.filenames.extend(glob.glob(os.path.join(images_dir, f"{str(i).zfill(5)}", "*.jpeg")))
                    self.filenames.extend(glob.glob(os.path.join(images_dir, f"{str(i).zfill(5)}", "*.jpg")))
                else:
                    self.filenames.extend(glob.glob(os.path.join(images_dir, f"{str(i).zfill(5)}", "*.png")))
                    self.filenames.extend(glob.glob(os.path.join(images_dir, f"{str(i).zfill(5)}", "*.jpeg")))
                    self.filenames.extend(glob.glob(os.path.join(images_dir, f"{str(i).zfill(5)}", "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform
        self.file_name_to_fingerprint = {}
        with open(note_file) as f:
            for line in f:
                filename, fingerprint = line.split()
                self.file_name_to_fingerprint[filename.split("/")[-1]] = fingerprint
            

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        fingerprint = self.file_name_to_fingerprint[filename.split("/")[-1]]
        fingerprint = torch.tensor([int(x) for x in fingerprint]).int()
        image = PIL.Image.open(filename).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, fingerprint

    def __len__(self):
        return len(self.filenames)


def load_decoder():
    global RevealNet
    global FINGERPRINT_SIZE

    from models import StegaStampDecoder
    state_dict = torch.load(args.decoder_path)
    FINGERPRINT_SIZE = state_dict["dense.2.weight"].shape[0]

    RevealNet = StegaStampDecoder(args.image_resolution, 3, FINGERPRINT_SIZE)
    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    RevealNet.load_state_dict(torch.load(args.decoder_path, **kwargs))
    RevealNet = RevealNet.to(device)


def load_data():
    global dataset, dataloader

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    s = time()
    print(f"Loading image folder {args.images_dir} and note file {args.note_file} ...")
    dataset = CustomImageFolder(args.images_dir, args.note_file, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")


def extract_fingerprints():
    all_fingerprinted_images = []
    bitwise_accuracy = 0

    BATCH_SIZE = args.batch_size

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_predicted_fingerprints = []
    for images, fingerprints in tqdm(dataloader):
        images = images.to(device)
        fingerprints = fingerprints.to(device)

        predicted_fingerprints = RevealNet(images)
        predicted_fingerprints = (predicted_fingerprints > 0).long()

        bitwise_accuracy += (fingerprints.detach() == predicted_fingerprints).float().mean(dim=1).sum().item()

        all_predicted_fingerprints.append(predicted_fingerprints.detach().cpu())

    all_predicted_fingerprints = torch.cat(all_predicted_fingerprints, dim=0)
    bitwise_accuracy = bitwise_accuracy / len(all_predicted_fingerprints)
    print(f"Bitwise accuracy on fingerprinted images: {bitwise_accuracy}") # non-corrected


def main():
    
    load_decoder()
    load_data()
    extract_fingerprints()


if __name__ == "__main__":
    main()
