import torch
import torch.nn as nn

class GANLoss(nn.Module):
    """GAN loss for adversarial training"""
    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class CycleLoss(nn.Module):
    """Cycle consistency loss"""
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
    
    def __call__(self, real_A, cycled_A, real_B, cycled_B):
        loss_A = self.criterion(cycled_A, real_A)
        loss_B = self.criterion(cycled_B, real_B)
        return loss_A + loss_B

class IdentityLoss(nn.Module):
    """Identity loss for CycleGAN"""
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
    
    def __call__(self, real, same):
        return self.criterion(same, real)
