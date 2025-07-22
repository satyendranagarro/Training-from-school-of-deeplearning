import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    return psnr(img1, img2, data_range=2.0)  # Range [-1, 1]

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    # Handle multi-channel images
    if len(img1.shape) == 3:
        ssim_values = []
        for i in range(img1.shape[0]):
            ssim_val = ssim(img1[i], img2[i], data_range=2.0)
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:
        return ssim(img1, img2, data_range=2.0)

def calculate_ndvi(nir_band, red_band):
    """Calculate NDVI from NIR and Red bands"""
    if isinstance(nir_band, torch.Tensor):
        nir_band = nir_band.cpu().numpy()
    if isinstance(red_band, torch.Tensor):
        red_band = red_band.cpu().numpy()
    
    # Denormalize from [-1, 1] to [0, 1]
    nir = (nir_band + 1) / 2
    red = (red_band + 1) / 2
    
    # Calculate NDVI
    ndvi = (nir - red) / (nir + red + 1e-8)
    return ndvi

def calculate_mae(img1, img2):
    """Calculate Mean Absolute Error"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    return np.mean(np.abs(img1 - img2))

def calculate_rmse(img1, img2):
    """Calculate Root Mean Square Error"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    return np.sqrt(np.mean((img1 - img2) ** 2))

def evaluate_model(model, test_loader, config_name):
    """Evaluate model performance on test set"""
    model.eval_mode()
    
    metrics = {
        'psnr': [],
        'ssim': [],
        'mae': [],
        'rmse': [],
        'ndvi_corr': []
    }
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            model.set_input(batch)
            model.forward()
            
            # Get generated and real images
            fake_B = model.fake_B.cpu()  # Generated EO
            real_B = model.real_B.cpu()  # Real EO
            
            # Calculate metrics for each sample in batch
            for j in range(fake_B.shape[0]):
                fake_img = fake_B[j]
                real_img = real_B[j]
                
                # PSNR and SSIM
                psnr_val = calculate_psnr(fake_img, real_img)
                ssim_val = calculate_ssim(fake_img, real_img)
                mae_val = calculate_mae(fake_img, real_img)
                rmse_val = calculate_rmse(fake_img, real_img)
                
                metrics['psnr'].append(psnr_val)
                metrics['ssim'].append(ssim_val)
                metrics['mae'].append(mae_val)
                metrics['rmse'].append(rmse_val)
                
                # NDVI correlation (if applicable)
                if config_name in ['rgb', 'rgb_nir'] and fake_img.shape[0] >= 3:
                    # Assume NIR is at index 3 for rgb_nir, or use red for rgb
                    if config_name == 'rgb_nir' and fake_img.shape[0] >= 4:
                        nir_idx = 3
                        red_idx = 0  # Red band
                        
                        fake_ndvi = calculate_ndvi(fake_img[nir_idx], fake_img[red_idx])
                        real_ndvi = calculate_ndvi(real_img[nir_idx], real_img[red_idx])
                        
                        # Calculate correlation
                        corr = np.corrcoef(fake_ndvi.flatten(), real_ndvi.flatten())[0, 1]
                        if not np.isnan(corr):
                            metrics['ndvi_corr'].append(corr)
    
    # Calculate mean metrics
    mean_metrics = {}
    for key, values in metrics.items():
        if values:
            mean_metrics[key] = np.mean(values)
            mean_metrics[f'{key}_std'] = np.std(values)
        else:
            mean_metrics[key] = 0.0
            mean_metrics[f'{key}_std'] = 0.0
    
    return mean_metrics

def print_metrics(metrics, config_name):
    """Print evaluation metrics in a formatted way"""
    print(f"\n=== Evaluation Results for {config_name.upper()} ===")
    print(f"PSNR: {metrics['psnr']:.4f} ± {metrics['psnr_std']:.4f}")
    print(f"SSIM: {metrics['ssim']:.4f} ± {metrics['ssim_std']:.4f}")
    print(f"MAE: {metrics['mae']:.4f} ± {metrics['mae_std']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f} ± {metrics['rmse_std']:.4f}")
    
    if metrics['ndvi_corr'] > 0:
        print(f"NDVI Correlation: {metrics['ndvi_corr']:.4f} ± {metrics['ndvi_corr_std']:.4f}")
    
    print("=" * 50)
