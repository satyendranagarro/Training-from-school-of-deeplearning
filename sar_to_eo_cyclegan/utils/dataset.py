import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from .preprocessing import preprocess_sar_image, preprocess_sentinel2_image, normalize_to_gan_range

class Sen12MSDataset(Dataset):
    """Dataset class for Sen12MS SAR-to-EO translation"""
    
    def __init__(self, sar_dir, eo_dir, eo_bands, transform=None):
        self.sar_dir = sar_dir
        self.eo_dir = eo_dir
        self.eo_bands = eo_bands
        self.transform = transform
        print("hi")
        # Get matching SAR and EO file pairs
        self.file_pairs = self._get_file_pairs()
        print(f"Found {len(self.file_pairs)} valid file pairs")
        
    def _get_file_pairs(self):
        """Match SAR and EO files based on naming convention"""
        sar_files = glob.glob(os.path.join(self.sar_dir, "**/*.tif"), recursive=True)
        pairs = []
        print(len(sar_files)) 

        for sar_file in sar_files[0:550]:
            # Extract identifier from SAR filename
            sar_basename = os.path.basename(sar_file)
            sar_id = sar_basename.split('_')[0]
            
            # Find corresponding EO file
            eo_pattern = os.path.join(self.eo_dir, "**", f"{sar_id}*.tif")
            eo_matches = glob.glob(eo_pattern, recursive=True)
            
            if eo_matches:
                pairs.append((sar_file, eo_matches[0]))
        
        return pairs

    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        sar_path, eo_path = self.file_pairs[idx]
        
        try:
            # Load and preprocess SAR image
            sar_data = preprocess_sar_image(sar_path)
            if sar_data is None:
                return self.__getitem__((idx + 1) % len(self.file_pairs))
            
            sar_normalized, sar_params = normalize_to_gan_range(sar_data)
            
            # Load and preprocess EO image
            eo_data = preprocess_sentinel2_image(eo_path, self.eo_bands)
            if eo_data is None:
                return self.__getitem__((idx + 1) % len(self.file_pairs))
            
            eo_normalized, eo_params = normalize_to_gan_range(eo_data)
            
            # Convert to PyTorch tensors
            sar_tensor = torch.FloatTensor(sar_normalized)
            eo_tensor = torch.FloatTensor(eo_normalized)
            
            return {
                'sar': sar_tensor,
                'eo': eo_tensor,
                'sar_params': sar_params,
                'eo_params': eo_params,
                'sar_path': sar_path,
                'eo_path': eo_path
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self.file_pairs))

def create_data_loader(dataset, batch_size=4, num_workers=4, shuffle=True):
    """Create data loader for training/testing"""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

def validate_dataset(dataset, num_samples=5):
    """Validate dataset quality"""
    print(f"Dataset size: {len(dataset)}")
    
    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            sar_tensor = sample['sar']
            eo_tensor = sample['eo']
            
            print(f"Sample {i}:")
            print(f"  SAR range: [{sar_tensor.min():.3f}, {sar_tensor.max():.3f}]")
            print(f"  EO range: [{eo_tensor.min():.3f}, {eo_tensor.max():.3f}]")
            print(f"  SAR shape: {sar_tensor.shape}")
            print(f"  EO shape: {eo_tensor.shape}")
            
            # Check for invalid values
            if torch.any(torch.isnan(sar_tensor)) or torch.any(torch.isinf(sar_tensor)):
                print(f"  WARNING: SAR contains invalid values")
            if torch.any(torch.isnan(eo_tensor)) or torch.any(torch.isinf(eo_tensor)):
                print(f"  WARNING: EO contains invalid values")
                
        except Exception as e:
            print(f"Error validating sample {i}: {e}")
