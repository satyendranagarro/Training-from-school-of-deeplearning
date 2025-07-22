import numpy as np
import rasterio
from scipy.ndimage import median_filter
import torch
import os
import pickle

def preprocess_sar_image(sar_path):
    """Load and preprocess Sentinel-1 SAR image with speckle reduction."""
    try:
        with rasterio.open(sar_path) as src:
            sar_data = src.read().astype(np.float32)  # Shape: [C, H, W]

        # Replace no-data values with NaN
        sar_data = np.where(sar_data == -32768, np.nan, sar_data)

        # Median filter (despeckle)
        for i in range(sar_data.shape[0]):
            sar_data[i] = median_filter(sar_data[i], size=3)

        # Replace NaNs with global mean
        sar_data = np.nan_to_num(sar_data, nan=np.nanmean(sar_data))

        return sar_data  # [C, H, W]
    except Exception as e:
        print(f"[Error] SAR preprocessing failed for {sar_path}: {e}")
        return None

def preprocess_sentinel2_image(s2_path, target_bands):
    """Load and preprocess Sentinel-2 EO image for specific band indices."""
    try:
        with rasterio.open(s2_path) as src:
            s2_data = src.read().astype(np.float32)  # [C, H, W]

        # Select desired bands
        selected_bands = s2_data[target_bands]  # Shape: [C', H, W]

        # Mask invalid values (0 or saturated)
        selected_bands = np.where((selected_bands == 0) | (selected_bands > 10000), np.nan, selected_bands)

        # Fill NaNs with band-wise median
        for i in range(selected_bands.shape[0]):
            band = selected_bands[i]
            if np.any(np.isnan(band)):
                median_val = np.nanmedian(band)
                selected_bands[i] = np.nan_to_num(band, nan=median_val)

        return selected_bands
    except Exception as e:
        print(f"[Error] EO preprocessing failed for {s2_path}: {e}")
        return None

def normalize_to_gan_range(data):
    """Normalize image to [-1, 1] for GAN."""
    data = np.asarray(data).astype(np.float32)

    # Clip outliers
    p1, p99 = np.percentile(data, [1, 99])
    data_clipped = np.clip(data, p1, p99)

    # Min-max normalize to [0, 1]
    data_min, data_max = np.min(data_clipped), np.max(data_clipped)
    if data_max == data_min:
        normalized = np.zeros_like(data_clipped)
    else:
        normalized = (data_clipped - data_min) / (data_max - data_min)

    # Scale to [-1, 1]
    gan_scaled = 2 * normalized - 1

    return gan_scaled, (data_min, data_max)

def denormalize_from_gan_range(data, params):
    """Convert GAN-scaled image [-1, 1] back to original range."""
    data_min, data_max = params
    normalized = (data + 1) / 2  # Back to [0, 1]
    original = normalized * (data_max - data_min) + data_min
    return original

def save_normalization_params(sar_params, eo_params, config_name, save_dir):
    """Save normalization stats to .pkl for reproducibility."""
    os.makedirs(save_dir, exist_ok=True)
    params = {
        'sar_params': sar_params,
        'eo_params': eo_params,
        'config': config_name
    }

    filepath = os.path.join(save_dir, f'normalization_params_{config_name}.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(params, f)

    print(f"[Saved] Normalization params → {filepath}")

def load_normalization_params(config_name, save_dir):
    """Load pre-saved normalization stats."""
    filepath = os.path.join(save_dir, f'normalization_params_{config_name}.pkl')

    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"[Warning] No normalization params found: {filepath}")
        return None
