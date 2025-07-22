import torch
import torch.nn as nn
from .networks import define_G, define_D
from .losses import GANLoss, CycleLoss, IdentityLoss
import itertools
import os

class CycleGANModel:
    """CycleGAN model for SAR-to-EO translation"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

        sar_channels = config['model']['input_nc']  # 2 channels
        eo_channels = config['band_configs'][config['current_config']]['channels']
        
        # Model parameters
        input_nc = config['model']['input_nc']  # SAR channels
        self.output_nc = config['band_configs'][config['current_config']]['channels']  # EO channels
        ngf = config['model']['ngf']
        ndf = config['model']['ndf']
        norm = config['model']['norm']
        use_dropout = config['model']['use_dropout']
        init_type = config['model']['init_type']
        init_gain = config['model']['init_gain']
   
        # Training parameters
        self.lambda_cycle = config['training']['lambda_cycle']
        self.lambda_identity = config['training']['lambda_identity']
        
        # Initialize networks
        gpu_ids = [0] if torch.cuda.is_available() else []
        
        # Generators
        self.netG_A = define_G(input_nc, self.output_nc, ngf, norm, use_dropout, init_type, init_gain, gpu_ids)
        # self.netG_A = define_G(sar_channels, eo_channels, ngf, norm, use_dropout, init_type, init_gain, gpu_ids)
        self.netG_B = define_G(self.output_nc, input_nc, ngf, norm, use_dropout, init_type, init_gain, gpu_ids)
        # self.netG_B = define_G(eo_channels, sar_channels, ngf, norm, use_dropout, init_type, init_gain, gpu_ids)
        
        # Discriminators
        self.netD_A = define_D(input_nc, ndf, norm, init_type, init_gain, gpu_ids)
        self.netD_B = define_D(self.output_nc, ndf, norm, init_type, init_gain, gpu_ids)
        
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
        
        # Learning rate schedulers
        self.schedulers = []
        self.schedulers.append(torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, self.lr_lambda))
        self.schedulers.append(torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, self.lr_lambda))
        
    def lr_lambda(self, epoch):
        """Learning rate decay"""
        decay_start_epoch = 5
        total_epochs = 10
        if epoch < decay_start_epoch:
            return 1.0
        else:
            return 1.0 - max(0, epoch - decay_start_epoch) / (total_epochs - decay_start_epoch)
    
    def set_input(self, input_data):
        """Set input data for the model"""
        self.real_A = input_data['sar'].to(self.device)  # SAR images
        self.real_B = input_data['eo'].to(self.device)   # EO images
    
    def forward(self):
        """Forward pass through the networks"""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
    
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    
    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, self.fake_A)
    
    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, self.fake_B)
    
    def backward_G(self):
        """Calculate loss for generators"""
        lambda_idt = self.lambda_identity
        lambda_cycle = self.lambda_cycle
        
        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.real_B, self.idt_A) * lambda_idt
            
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.real_A, self.idt_B) * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_B(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_A(self.fake_A), True)
        
        # Cycle loss
        self.loss_cycle = self.criterionCycle(self.real_A, self.rec_A, self.real_B, self.rec_B) * lambda_cycle
        
        # Combined generator loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()
    
    def optimize_parameters(self):
        """Optimize network parameters"""
        # Forward pass
        self.forward()
        
        # Update generators
        self.netD_A.requires_grad_(False)
        self.netD_B.requires_grad_(False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        # Update discriminators
        self.netD_A.requires_grad_(True)
        self.netD_B.requires_grad_(True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
    
    def get_current_losses(self):
        """Return current losses"""
        errors_ret = {}
        for name in ['D_A', 'G_A', 'cycle', 'idt_A', 'D_B', 'G_B', 'idt_B']:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret
    
    def update_learning_rate(self):
        """Update learning rates for all optimizers"""
        for scheduler in self.schedulers:
            scheduler.step()
    
    def save_networks(self, epoch, save_dir):
        """Save network weights"""
        os.makedirs(save_dir, exist_ok=True)
        
        save_filename = f'net_G_A_epoch_{epoch}.pth'
        save_path = os.path.join(save_dir, save_filename)
        torch.save(self.netG_A.state_dict(), save_path)
        
        save_filename = f'net_G_B_epoch_{epoch}.pth'
        save_path = os.path.join(save_dir, save_filename)
        torch.save(self.netG_B.state_dict(), save_path)
        
        save_filename = f'net_D_A_epoch_{epoch}.pth'
        save_path = os.path.join(save_dir, save_filename)
        torch.save(self.netD_A.state_dict(), save_path)
        
        save_filename = f'net_D_B_epoch_{epoch}.pth'
        save_path = os.path.join(save_dir, save_filename)
        torch.save(self.netD_B.state_dict(), save_path)
        
        print(f"Networks saved at epoch {epoch}")
    
    def load_networks(self, epoch, save_dir):
        """Load network weights"""
        load_filename = f'net_G_A_epoch_{epoch}.pth'
        load_path = os.path.join(save_dir, load_filename)
        self.netG_A.load_state_dict(torch.load(load_path))
        
        load_filename = f'net_G_B_epoch_{epoch}.pth'
        load_path = os.path.join(save_dir, load_filename)
        self.netG_B.load_state_dict(torch.load(load_path))
        
        print(f"Networks loaded from epoch {epoch}")
    
    def eval_mode(self):
        """Set networks to evaluation mode"""
        self.netG_A.eval()
        self.netG_B.eval()
        self.netD_A.eval()
        self.netD_B.eval()
    
    def train_mode(self):
        """Set networks to training mode"""
        self.netG_A.train()
        self.netG_B.train()
        self.netD_A.train()
        self.netD_B.train()