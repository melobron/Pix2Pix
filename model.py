import torch
import torch.nn as nn


######################################## GAN Loss ########################################
class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.register_buffer('real_target', torch.tensor(1.0))
        self.register_buffer('fake_target', torch.tensor(0.0))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, input_tensor, target_is_real):
        if target_is_real:
            target_tensor = self.real_target
        else:
            target_tensor = self.fake_target
        return target_tensor.expand_as(input_tensor)

    def __call__(self, input_tensor, target_is_real):
        target_tensor = self.get_target_tensor(input_tensor, target_is_real)
        return self.loss(input_tensor, target_tensor)


######################################## UNet Generator ########################################
class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, n_downs=8, n_feats=64, norm_layer=nn.BatchNorm2d):
        super(UNetGenerator, self).__init__()

        # Construct UNet from the innermost to the outermost block
        unet_block = UNetSkipConnectionBlock(n_feats*8, n_feats*8, input_channels=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for _ in range(n_downs-5):
            unet_block = UNetSkipConnectionBlock(n_feats*8, n_feats*8, input_channels=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSkipConnectionBlock(n_feats*4, n_feats*8, input_channels=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSkipConnectionBlock(n_feats*2, n_feats*4, input_channels=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSkipConnectionBlock(n_feats, n_feats*2, input_channels=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UNetSkipConnectionBlock(out_channels, n_feats, input_channels=in_channels, submodule=unet_block, norm_layer=norm_layer, outermost=True)

    def forward(self, x):
        return self.model(x)


class UNetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_channels, inner_channels, input_channels=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d):
        super(UNetSkipConnectionBlock, self).__init__()

        self.outermost = outermost

        if input_channels is None:
            input_channels = outer_channels
        downconv = nn.Conv2d(input_channels, inner_channels, 4, 2, 1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_channels)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_channels)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_channels*2, outer_channels, 4, 2, 1, bias=False)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_channels, outer_channels, 4, 2, 1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_channels*2, outer_channels, 4, 2, 1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], dim=1)


######################################## Resnet Generator ########################################
class ResnetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, n_feats, n_blocks=9):
        super(ResnetGenerator, self).__init__()

        in_conv = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, n_feats, 7, 1, 0, bias=False),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(True),
        ]
        self.in_conv = nn.Sequential(*in_conv)

        down = [
            nn.Conv2d(n_feats, n_feats*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(n_feats*2),
            nn.ReLU(True),
            nn.Conv2d(n_feats*2, n_feats*4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(n_feats*4),
            nn.ReLU(True),
        ]
        self.down = nn.Sequential(*down)

        body = []
        for i in range(n_blocks):
            body += [ResBlock(n_feats=n_feats*4)]
        self.body = nn.Sequential(*body)

        up = [
            nn.ConvTranspose2d(n_feats*4, n_feats*2, 3, 2, 1, output_padding=1, bias=False),
            nn.BatchNorm2d(n_feats*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_feats*2, n_feats, 3, 2, 1, output_padding=1, bias=False),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(True),
        ]
        self.up = nn.Sequential(*up)

        out_conv = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_feats, out_channels, 7, 1, 0),
            nn.Tanh()
        ]
        self.out_conv = nn.Sequential(*out_conv)

    def forward(self, x):
        out = self.in_conv(x)
        out = self.down(out)
        out = self.body(out)
        out = self.up(out)
        return self.out_conv(out)


class ResBlock(nn.Module):
    def __init__(self, n_feats):
        super(ResBlock, self).__init__()

        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_feats, n_feats, 3, 1, 0, bias=False),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_feats, n_feats, 3, 1, 0, bias=False),
            nn.BatchNorm2d(n_feats),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        out = x + self.body(x)
        return nn.ReLU(True)(out)


######################################## Discriminator ########################################
class Discriminator(nn.Module):
    def __init__(self, in_channels, n_feats, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()

        body = [
            nn.Conv2d(in_channels, n_feats, 4, 2, 1),
            nn.LeakyReLU(0.2, True)
        ]

        in_c, out_c = n_feats, n_feats
        for n in range(1, n_layers):
            in_c = out_c
            out_c = min(2**n, 8) * n_feats
            body += [
                nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
                norm_layer(out_c),
                nn.LeakyReLU(0.2, True)
            ]

        in_c = out_c
        out_c = min(2**n_layers, 8)
        body += [
            nn.Conv2d(in_c, out_c, 4, 1, 1, bias=False),
            norm_layer(out_c),
            nn.LeakyReLU(0.2, True)
        ]

        body += [nn.Conv2d(out_c, 1, 4, 1, 1)]
        self.model = nn.Sequential(*body)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    resnet = ResnetGenerator(in_channels=3, out_channels=3, n_feats=64)
    discriminator = Discriminator(in_channels=6, n_feats=64, n_layers=3, norm_layer=nn.BatchNorm2d)

    input = torch.randn(10, 3, 256, 256)
    print('resnet input:{}'.format(input.shape))
    output = resnet(input)
    print('resnet output:{}'.format(output.shape))

    input = torch.randn(10, 6, 256, 256)
    print('discriminator input:{}'.format(input.shape))
    output = discriminator(input)
    print('discriminator output:{}'.format(output.shape))
