import numpy as np
import rasterio
from scipy.ndimage import median_filter
import torch
import os
import pickle

def preprocess_sar_image(sar_path):
    """Preprocess Sentinel-1 SAR image"""
    try:
        with rasterio.open(sar_path) as src:
            sar_data = src.read().astype(np.float32)
        
        # Handle invalid values
        sar_data = np.where(sar_data == -32768, np.nan, sar_data)
        
        # Apply speckle filtering
        for i in range(sar_data.shape[0]):
            sar_data[i] = median_filter(sar_data[i], size=3)
        
        # Handle NaN values
        sar_data = np.nan_to_num(sar_data, nan=np.nanmean(sar_data))
        
        return sar_data
    except Exception as e:
        print(f"Error processing SAR image {sar_path}: {e}")
        return None

def preprocess_sentinel2_image(s2_path, target_bands):
    """Preprocess Sentinel-2 image for specific band configuration"""
    try:
        with rasterio.open(s2_path) as src:
            s2_data = src.read().astype(np.float32)
        
        # Select target bands
        selected_bands = s2_data[target_bands]
        
        # Handle invalid values
        selected_bands = np.where(selected_bands == 0, np.nan, selected_bands)
        selected_bands = np.where(selected_bands > 10000, np.nan, selected_bands)
        
        # Fill NaN values with band-wise median
        for i in range(selected_bands.shape[0]):
            band_data = selected_bands[i]
            if np.any(np.isnan(band_data)):
                median_val = np.nanmedian(band_data)
                selected_bands[i] = np.nan_to_num(band_data, nan=median_val)
        
        return selected_bands
    except Exception as e:
        print(f"Error processing EO image {s2_path}: {e}")
        return None

def normalize_to_gan_range(data):
    """Normalize data to [-1, 1] range for GAN training"""
    # Clip extreme values
    p1, p99 = np.percentile(data, [1, 99])
    data_clipped = np.clip(data, p1, p99)
    
    # Min-max normalization
    data_min = np.min(data_clipped)
    data_max = np.max(data_clipped)
    
    if data_max == data_min:
        normalized = np.zeros_like(data_clipped)
    else:
        normalized = (data_clipped - data_min) / (data_max - data_min)
    
    # Scale to [-1, 1]
    gan_ready = 2 * normalized - 1
    
    return gan_ready, (data_min, data_max)

def denormalize_from_gan_range(data, params):
    """Denormalize data from [-1, 1] range back to original scale"""
    data_min, data_max = params
    
    # Scale from [-1, 1] to [0, 1]
    normalized = (data + 1) / 2
    
    # Scale back to original range
    original = normalized * (data_max - data_min) + data_min
    
    return original

def save_normalization_params(sar_params, eo_params, config_name, save_dir):
    """Save normalization parameters for post-processing"""
    os.makedirs(save_dir, exist_ok=True)
    params = {
        'sar_params': sar_params,
        'eo_params': eo_params,
        'config': config_name
    }
    
    filepath = os.path.join(save_dir, f'normalization_params_{config_name}.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(params, f)
    
    print(f"Normalization parameters saved to {filepath}")

def load_normalization_params(config_name, save_dir):
    """Load normalization parameters"""
    filepath = os.path.join(save_dir, f'normalization_params_{config_name}.pkl')
    
    try:
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        return params
    except FileNotFoundError:
        print(f"Normalization parameters not found at {filepath}")
        return None