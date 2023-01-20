import argparse
from train import TrainPix2Pix

# Arguments
parser = argparse.ArgumentParser(description='Train Pix2Pix')

parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--seed', type=int, default=100)

# Training parameters
parser.add_argument('--n_epochs', type=int, default=200)  # initial_lr_epoch + decay_lr_epoch
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=2e-4)

# Scheduler parameters
parser.add_argument('--scheduler', type=str, default='linear')
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--initial_lr_epoch', type=int, default=100)
parser.add_argument('--decay_lr_epoch', type=int, default=100)
parser.add_argument('--decay_lr_step', type=int, default=100)

# Weighted Loss
parser.add_argument('--lambda_L1', type=float, default=10.0)

# Model
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=3)
parser.add_argument('--n_downs', type=int, default=3)
parser.add_argument('--n_feats', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--init_type', type=str, default='normal')
parser.add_argument('--init_gain', type=float, default=0.02)

# Dataset
parser.add_argument('--domain1', type=str, default='photo')
parser.add_argument('--domain2', type=str, default='segmentation')

# Transformations
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--crop', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--flip_rotate', type=bool, default=True)
parser.add_argument('--normalize', type=bool, default=True)

args = parser.parse_args()

train_Pix2Pix = TrainPix2Pix(args)
train_Pix2Pix.train()
