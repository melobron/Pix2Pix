import cv2
import torch.optim as optim
from model import UNetGenerator
from utils import *
import argparse
from datasets import Facades
import matplotlib.pyplot as plt

# Arguments
parser = argparse.ArgumentParser(description='Train CycleGAN')
# Dataset
parser.add_argument('--domain1', type=str, default='photo')
parser.add_argument('--domain2', type=str, default='segmentation')

# Transformations
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--flip_rotate', type=bool, default=True)
parser.add_argument('--normalize', type=bool, default=True)
args = parser.parse_args()

transform = A.Compose(get_transforms(args), additional_targets={'target': 'image'})

dataset = Facades.FacadesDataset(domain1=args.domain1, domain2=args.domain2, train=True, transform=transform)
domain1_tensor, domain2_tensor = dataset[0]['domain1'], dataset[0]['domain2']
domain1_tensor, domain2_tensor = 0.5*(domain1_tensor+1), 0.5*(domain2_tensor+1)
domain1_numpy = tensor_to_numpy(domain1_tensor)
domain2_numpy = tensor_to_numpy(domain2_tensor)

plt.imshow(domain1_numpy)
plt.show()
