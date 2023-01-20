import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import json
import pandas as pd
import matplotlib.pyplot as plt

from model import ResnetGenerator, Discriminator, GANLoss
from datasets import Facades
from utils import *


class TrainPix2Pix:
    def __init__(self, args):
        # Arguments
        self.args = args

        # Device
        self.gpu_num = args.gpu_num
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

        # Random Seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        else:
            torch.manual_seed(args.seed)

        # Models
        self.net_G = ResnetGenerator(args.in_channels, args.out_channels, args.n_feats, n_blocks=9).to(self.device)
        self.net_D = Discriminator(args.in_channels+args.out_channels, args.n_feats, args.n_layers).to(self.device)

        init_weights(self.net_G, args.init_type, args.init_gain)
        init_weights(self.net_D, args.init_type, args.init_gain)

        # Training Parameters
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.lambda_L1 = args.lambda_L1

        # Loss
        self.criterion_GAN = GANLoss().to(self.device)
        self.criterion_L1 = nn.L1Loss().to(self.device)
        self.criterion_MSE = nn.MSELoss().to(self.device)

        # Optimizer
        self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.net_D.parameters(), lr=args.lr, betas=(0.5, 0.999))

        # Scheduler
        self.scheduler_G = get_scheduler(self.optimizer_G, args)
        self.scheduler_D = get_scheduler(self.optimizer_D, args)

        # Transform
        transform = A.Compose(get_transforms(args), additional_targets={'target': 'image'})

        # Dataset
        self.dataset = Facades.FacadesDataset(domain1=args.domain1, domain2=args.domain2, train=True, transform=transform)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True)

        # Save Paths
        self.exp_dir, self.exp_num = make_exp_dir('./experiments/')['new_dir'], make_exp_dir('./experiments/')['new_dir_num']
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.result_dir = os.path.join(self.exp_dir, 'results')
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # Save Argument file
        param_file = os.path.join(self.exp_dir, 'params.json')
        with open(param_file, mode='w') as f:
            json.dump(args.__dict__, f, indent=4)

    def train(self):
        print(self.device)

        # Losses
        G_total_losses, G_GAN_losses, G_L1_losses = [], [], []
        D_total_losses, D_fake_losses, D_real_losses = [], [], []

        start = time.time()
        for epoch in range(1, self.n_epochs + 1):

            # Training
            for batch, data in enumerate(self.dataloader):
                real_1, real_2 = data['domain1'], data['domain2']
                real_1, real_2 = real_1.to(self.device), real_2.to(self.device)
                fake_2 = self.net_G(real_1)

                # Update Discriminator
                self.optimizer_D.zero_grad()

                fake_12 = torch.cat([real_1, fake_2], dim=1)
                pred_fake = self.net_D(fake_12.detach())
                loss_d_fake = self.criterion_GAN(pred_fake, False)

                real_12 = torch.cat([real_1, real_2], dim=1)
                pred_real = self.net_D(real_12)
                loss_d_real = self.criterion_GAN(pred_real, True)

                loss_d = (loss_d_fake + loss_d_real) * 0.5
                loss_d.backward()
                self.optimizer_D.step()

                # Update Generator
                self.optimizer_G.zero_grad()

                fake_12 = torch.cat([real_1, fake_2], dim=1)
                pred_fake = self.net_D(fake_12)
                loss_g_gan = self.criterion_GAN(pred_fake, True)

                loss_g_l1 = self.criterion_L1(fake_2, real_2)

                loss_g = loss_g_gan + loss_g_l1*self.lambda_L1
                loss_g.backward()
                self.optimizer_G.step()

                print('[Epoch {}][{}/{}] | loss: G_total={:.3f} G_GAN={:.3f} G_L1={:.3f} D_total={:.3f}'.format(
                    epoch, (batch + 1) * self.batch_size, len(self.dataset),
                    loss_g.item(), loss_g_gan.item(), loss_g_l1.item(), loss_d.item()
                ))

                # Save Losses
                G_total_losses.append(loss_g.item())
                G_GAN_losses.append(loss_g_gan.item())
                G_L1_losses.append(loss_g_l1.item())
                D_total_losses.append(loss_d.item())
                D_fake_losses.append(loss_d_fake.item())
                D_real_losses.append(loss_d_real.item())

            update_lr(self.scheduler_G, self.optimizer_G)
            update_lr(self.scheduler_D, self.optimizer_D)

            # Checkpoints
            if epoch % 100 == 0 or epoch == self.n_epochs:
                torch.save(self.net_G.state_dict(), os.path.join(self.checkpoint_dir, 'netG_{}epochs.pth'.format(epoch)))
                torch.save(self.net_D.state_dict(), os.path.join(self.checkpoint_dir, 'netD_{}epochs.pth'.format(epoch)))

        # Visualize Loss
        fig, axs = plt.subplots(3, 2)
        axs[0, 0].plot(pd.DataFrame(G_total_losses))
        axs[0, 0].set_title('G Total Loss')
        axs[0, 1].plot(pd.DataFrame(G_GAN_losses))
        axs[0, 1].set_title('G GAN Loss')
        axs[1, 0].plot(pd.DataFrame(G_L1_losses))
        axs[1, 0].set_title('G L1 Loss')
        axs[1, 1].plot(pd.DataFrame(D_total_losses))
        axs[1, 1].set_title('D Total Loss')
        axs[2, 0].plot(pd.DataFrame(D_real_losses))
        axs[2, 0].set_title('D Real Loss')
        axs[2, 1].plot(pd.DataFrame(D_fake_losses))
        axs[2, 1].set_title('D Fake Loss')

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'evaluation.png'))
        plt.show()
