import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from .preprocessing import preprocess_sar_image, preprocess_sentinel2_image, normalize_to_gan_range

class Sen12MSDataset(Dataset):
    """
    Dataset class for SAR-to-EO translation using Sen12MS.
    """

    def __init__(self, sar_dir, eo_dir, eo_bands, transform=None):
        self.sar_dir = sar_dir
        self.eo_dir = eo_dir
        self.eo_bands = eo_bands
        self.transform = transform

        self.file_pairs = self._get_file_pairs()
        print(f"[Dataset] Found {len(self.file_pairs)} valid SAR-EO file pairs.")

    def _get_file_pairs(self):
        sar_files = glob.glob(os.path.join(self.sar_dir, "**/*.tif"), recursive=True)
        sar_files = sorted(sar_files)[:550]  # limit for training

        pairs = []
        for sar_path in sar_files:
            sar_id = os.path.basename(sar_path).split('_')[0]
            eo_pattern = os.path.join(self.eo_dir, "**", f"{sar_id}*.tif")
            eo_matches = glob.glob(eo_pattern, recursive=True)
            if eo_matches:
                pairs.append((sar_path, eo_matches[0]))
        return pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        sar_path, eo_path = self.file_pairs[idx]

        try:
            sar_raw = preprocess_sar_image(sar_path)
            if sar_raw is None:
                return self.__getitem__((idx + 1) % len(self.file_pairs))
            sar_norm, sar_params = normalize_to_gan_range(sar_raw)

            eo_raw = preprocess_sentinel2_image(eo_path, self.eo_bands)
            if eo_raw is None:
                return self.__getitem__((idx + 1) % len(self.file_pairs))
            eo_norm, eo_params = normalize_to_gan_range(eo_raw)

            # Convert [H, W, C] to [C, H, W]
            sar_tensor = torch.FloatTensor(sar_norm).permute(2, 0, 1)
            eo_tensor = torch.FloatTensor(eo_norm).permute(2, 0, 1)

            return {
                'sar': sar_tensor,
                'eo': eo_tensor,
                'sar_params': sar_params,
                'eo_params': eo_params,
                'sar_path': sar_path,
                'eo_path': eo_path
            }

        except Exception as e:
            print(f"[Dataset] Error loading sample {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self.file_pairs))

def create_data_loader(dataset, batch_size=4, num_workers=2, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

def validate_dataset(dataset, num_samples=5):
    print(f"[Validation] Dataset size: {len(dataset)}")
    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            sar_tensor = sample['sar']
            eo_tensor = sample['eo']

            print(f"Sample {i}:")
            print(f"  SAR shape: {sar_tensor.shape} | Range: [{sar_tensor.min():.3f}, {sar_tensor.max():.3f}]")
            print(f"  EO shape: {eo_tensor.shape}   | Range: [{eo_tensor.min():.3f}, {eo_tensor.max():.3f}]")

        except Exception as e:
            print(f"[Validation] Error in sample {i}: {e}")
