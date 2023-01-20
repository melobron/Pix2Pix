import argparse
import cv2
import torch

from model import ResnetGenerator
from datasets import Facades
from utils import *

# Arguments
parser = argparse.ArgumentParser(description='Test CycleGAN')

parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--exp_num', type=int, default=6)

# Training parameters
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=1)

# Model
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=3)
parser.add_argument('--n_downs', type=int, default=3)
parser.add_argument('--n_feats', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=3)

# Dataset
parser.add_argument('--domain1', type=str, default='segmentation')
parser.add_argument('--domain2', type=str, default='photo')

# Transformations
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--crop', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--flip_rotate', type=bool, default=False)
parser.add_argument('--normalize', type=bool, default=True)

opt = parser.parse_args()


def TestPix2Pix(args):
    device = torch.device('cuda:{}'.format(args.gpu_num))

    # Random Seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    # Models
    net_G = ResnetGenerator(args.in_channels, args.out_channels, args.n_feats, n_blocks=9).to(device)
    net_G.load_state_dict(torch.load('./experiments/exp{}/checkpoints/netG_{}epochs.pth'.format(args.exp_num, args.n_epochs), map_location=device))
    net_G.eval()

    # Transform
    transform = A.Compose(get_transforms(args), additional_targets={'target': 'image'})

    # Dataset
    test_dataset = Facades.FacadesDataset(domain1=args.domain1, domain2=args.domain2, train=True, transform=transform)

    # Evaluate
    save_dir = './experiments/exp{}/results/{}epochs'.format(args.exp_num, args.n_epochs)
    domain1to2_dir = os.path.join(save_dir, '{}2{}'.format(args.domain1, args.domain2))
    if not os.path.exists(domain1to2_dir):
        os.makedirs(domain1to2_dir)

    for index, data in zip(range(5), test_dataset):
        real_1, real_2 = data['domain1'], data['domain2']
        real_1, real_2 = real_1.to(device), real_2.to(device)
        real_1, real_2 = torch.unsqueeze(real_1, dim=0), torch.unsqueeze(real_2, dim=0)

        fake_2 = net_G(real_1)
        comparison = torch.cat([real_1, real_2, fake_2], dim=3)
        comparison = 0.5*(comparison + 1.0)

        comparison = torch.squeeze(comparison).cpu()
        comparison = tensor_to_numpy(comparison)
        comparison = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(domain1to2_dir, '{}.png'.format(index+1)), comparison)


if __name__ == "__main__":
    TestPix2Pix(args=opt)
