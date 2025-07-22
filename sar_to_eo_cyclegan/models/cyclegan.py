import torch
import torch.nn as nn
from .custom_generator import SmallGenerator
from .custom_discriminator import SmallDiscriminator
from .losses import GANLoss, CycleLoss, IdentityLoss
import itertools
import os

class CycleGANModel:
    """CycleGAN model for SAR-to-EO translation using small custom networks"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

        sar_channels = config['model']['input_nc']  # 2 channels
        eo_channels = config['band_configs'][config['current_config']]['channels']

        input_nc = sar_channels
        self.output_nc = eo_channels

        # Training parameters
        self.lambda_cycle = config['training']['lambda_cycle']
        self.lambda_identity = config['training']['lambda_identity']

        # Initialize networks
        self.netG_A = SmallGenerator(input_nc, self.output_nc).to(self.device)
        self.netG_B = SmallGenerator(self.output_nc, input_nc).to(self.device)
        self.netD_A = SmallDiscriminator(input_nc).to(self.device)
        self.netD_B = SmallDiscriminator(self.output_nc).to(self.device)

        # Loss functions
        self.criterionGAN = GANLoss('lsgan').to(self.device)
        self.criterionCycle = CycleLoss().to(self.device)
        self.criterionIdt = IdentityLoss().to(self.device)

        # Optimizers
        lr = config['training']['lr']
        beta1 = config['training']['beta1']

        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
            lr=lr, betas=(beta1, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
            lr=lr, betas=(beta1, 0.999)
        )

        self.schedulers = [
            torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, self.lr_lambda),
            torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, self.lr_lambda)
        ]

    def lr_lambda(self, epoch):
        decay_start_epoch = 5
        total_epochs = 10
        if epoch < decay_start_epoch:
            return 1.0
        else:
            return 1.0 - max(0, epoch - decay_start_epoch) / (total_epochs - decay_start_epoch)

    def set_input(self, input_data):
        self.real_A = input_data['sar'].to(self.device)
        self.real_B = input_data['eo'].to(self.device)

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, self.fake_A)

    def backward_D_B(self):
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, self.fake_B)

    def backward_G(self):
        lambda_idt = self.lambda_identity
        lambda_cycle = self.lambda_cycle

        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.real_B, self.idt_A) * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.real_A, self.idt_B) * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_G_A = self.criterionGAN(self.netD_B(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_A(self.fake_A), True)
        self.loss_cycle = self.criterionCycle(self.real_A, self.rec_A, self.real_B, self.rec_B) * lambda_cycle
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.netD_A.requires_grad_(False)
        self.netD_B.requires_grad_(False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.netD_A.requires_grad_(True)
        self.netD_B.requires_grad_(True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

    def get_current_losses(self):
        return {
            name: float(getattr(self, 'loss_' + name)) for name in [
                'D_A', 'G_A', 'cycle', 'idt_A', 'D_B', 'G_B', 'idt_B']
        }

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def save_networks(self, epoch, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.netG_A.state_dict(), os.path.join(save_dir, f'net_G_A_epoch_{epoch}.pth'))
        torch.save(self.netG_B.state_dict(), os.path.join(save_dir, f'net_G_B_epoch_{epoch}.pth'))
        torch.save(self.netD_A.state_dict(), os.path.join(save_dir, f'net_D_A_epoch_{epoch}.pth'))
        torch.save(self.netD_B.state_dict(), os.path.join(save_dir, f'net_D_B_epoch_{epoch}.pth'))
        print(f"Networks saved at epoch {epoch}")

    def load_networks(self, epoch, save_dir):
        self.netG_A.load_state_dict(torch.load(os.path.join(save_dir, f'net_G_A_epoch_{epoch}.pth')))
        self.netG_B.load_state_dict(torch.load(os.path.join(save_dir, f'net_G_B_epoch_{epoch}.pth')))
        print(f"Networks loaded from epoch {epoch}")

    def eval_mode(self):
        self.netG_A.eval()
        self.netG_B.eval()
        self.netD_A.eval()
        self.netD_B.eval()

    def train_mode(self):
        self.netG_A.train()
        self.netG_B.train()
        self.netD_A.train()
        self.netD_B.train()