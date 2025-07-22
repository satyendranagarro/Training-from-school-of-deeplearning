import torch
import torch.nn as nn
import functools

class SmallGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SmallGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class SmallDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(SmallDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
    return net

def define_G(input_nc, output_nc, ngf, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = SmallGenerator(input_nc, output_nc)
    if len(gpu_ids) > 0:
        net.to(torch.device('cuda'))
        net = torch.nn.DataParallel(net, gpu_ids)
    return init_weights(net, init_type, init_gain)

def define_D(input_nc, ndf, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = SmallDiscriminator(input_nc)
    if len(gpu_ids) > 0:
        net.to(torch.device('cuda'))
        net = torch.nn.DataParallel(net, gpu_ids)
    return init_weights(net, init_type, init_gain)