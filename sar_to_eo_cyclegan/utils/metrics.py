import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    return psnr(img1, img2, data_range=2.0)  # Assuming range [-1, 1]

def calculate_ssim(img1, img2):
    """Calculate SSIM across channels and average."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    if len(img1.shape) == 3:  # Multi-channel: [C, H, W]
        return np.mean([
            ssim(img1[c], img2[c], data_range=2.0) for c in range(img1.shape[0])
        ])
    else:
        return ssim(img1, img2, data_range=2.0)

def calculate_mae(img1, img2):
    """Mean Absolute Error."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    return np.mean(np.abs(img1 - img2))

def calculate_rmse(img1, img2):
    """Root Mean Square Error."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    return np.sqrt(np.mean((img1 - img2) ** 2))

def calculate_ndvi(nir_band, red_band):
    """Calculate NDVI: (NIR - RED) / (NIR + RED). Input range: [-1, 1]."""
    if isinstance(nir_band, torch.Tensor):
        nir_band = nir_band.detach().cpu().numpy()
    if isinstance(red_band, torch.Tensor):
        red_band = red_band.detach().cpu().numpy()
    
    nir = (nir_band + 1) / 2
    red = (red_band + 1) / 2
    ndvi = (nir - red) / (nir + red + 1e-8)
    return ndvi

def evaluate_model(model, test_loader, config_name):
    """Evaluate model on test dataset."""
    model.eval_mode()
    
    metrics = {
        'psnr': [], 'ssim': [], 'mae': [], 'rmse': [], 'ndvi_corr': []
    }

    with torch.no_grad():
        for batch in test_loader:
            model.set_input(batch)
            model.forward()

            fake_B = model.fake_B.cpu()
            real_B = model.real_B.cpu()

            for i in range(fake_B.size(0)):
                fake = fake_B[i]
                real = real_B[i]

                metrics['psnr'].append(calculate_psnr(fake, real))
                metrics['ssim'].append(calculate_ssim(fake, real))
                metrics['mae'].append(calculate_mae(fake, real))
                metrics['rmse'].append(calculate_rmse(fake, real))

                # NDVI Correlation (optional)
                if config_name in ['rgb_nir'] and fake.shape[0] >= 4:
                    nir_idx = 3
                    red_idx = 0
                    fake_ndvi = calculate_ndvi(fake[nir_idx], fake[red_idx])
                    real_ndvi = calculate_ndvi(real[nir_idx], real[red_idx])
                    corr = np.corrcoef(fake_ndvi.flatten(), real_ndvi.flatten())[0, 1]
                    if not np.isnan(corr):
                        metrics['ndvi_corr'].append(corr)

    # Compute means and stds
    summary = {}
    for key, vals in metrics.items():
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            summary[key] = np.mean(vals)
            summary[f"{key}_std"] = np.std(vals)
        else:
            summary[key] = 0.0
            summary[f"{key}_std"] = 0.0

    return summary

def print_metrics(metrics, config_name):
    """Nicely format and display the metric output."""
    print(f"\n📊 Evaluation Metrics — [{config_name.upper()}]")
    print("-" * 50)
    print(f"PSNR         : {metrics['psnr']:.4f} ± {metrics['psnr_std']:.4f}")
    print(f"SSIM         : {metrics['ssim']:.4f} ± {metrics['ssim_std']:.4f}")
    print(f"MAE          : {metrics['mae']:.4f} ± {metrics['mae_std']:.4f}")
    print(f"RMSE         : {metrics['rmse']:.4f} ± {metrics['rmse_std']:.4f}")

    if 'ndvi_corr' in metrics and metrics['ndvi_corr'] > 0:
        print(f"NDVI Corr    : {metrics['ndvi_corr']:.4f} ± {metrics['ndvi_corr_std']:.4f}")
    print("=" * 50)
