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
            raise NotImplementedError(f"GAN mode '{gan_mode}' not implemented.")

    def get_target_tensor(self, prediction, target_is_real):
        return self.real_label.expand_as(prediction) if target_is_real else self.fake_label.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            return -prediction.mean() if target_is_real else prediction.mean()

class CycleLoss(nn.Module):
    """Cycle consistency loss"""
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, real_A, cycled_A, real_B, cycled_B):
        return self.criterion(cycled_A, real_A) + self.criterion(cycled_B, real_B)

class IdentityLoss(nn.Module):
    """Identity loss for preserving color composition"""
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, real, same):
        return self.criterion(same, real)
