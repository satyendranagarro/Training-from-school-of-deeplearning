import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from utils.dataset import Sen12MSDataset, create_data_loader
from utils.metrics import evaluate_model, print_metrics
from utils.preprocessing import denormalize_from_gan_range, load_normalization_params
from models.cyclegan import CycleGANModel

def visualize_results(sar_img, generated_eo, real_eo, save_path, config_name):
    """Visualize SAR input, generated EO, and real EO"""
    
    # Convert tensors to numpy
    if isinstance(sar_img, torch.Tensor):
        sar_img = sar_img.cpu().numpy()
    if isinstance(generated_eo, torch.Tensor):
        generated_eo = generated_eo.cpu().numpy()
    if isinstance(real_eo, torch.Tensor):
        real_eo = real_eo.cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # SAR image (VV channel)
    sar_display = (sar_img[0] + 1) / 2  # Normalize to [0, 1] for display
    axes[0].imshow(sar_display, cmap='gray')
    axes[0].set_title('SAR Input (VV)')
    axes[0].axis('off')
    
    # Generated EO
    if config_name == 'rgb' or config_name == 'rgb_nir':
        # RGB display
        eo_gen_rgb = generated_eo[:3].transpose(1, 2, 0)
        eo_gen_rgb = (eo_gen_rgb + 1) / 2  # Normalize to [0, 1]
        eo_gen_rgb = np.clip(eo_gen_rgb, 0, 1)
        axes[1].imshow(eo_gen_rgb)
    else:
        # Single channel display
        eo_gen_display = (generated_eo[0] + 1) / 2
        axes[1].imshow(eo_gen_display, cmap='RdYlGn')
    
    axes[1].set_title('Generated EO')
    axes[1].axis('off')
    
    # Real EO
    if config_name == 'rgb' or config_name == 'rgb_nir':
        # RGB display
        eo_real_rgb = real_eo[:3].transpose(1, 2, 0)
        eo_real_rgb = (eo_real_rgb + 1) / 2  # Normalize to [0, 1]
        eo_real_rgb = np.clip(eo_real_rgb, 0, 1)
        axes[2].imshow(eo_real_rgb)
    else:
        # Single channel display
        eo_real_display = (real_eo[0] + 1) / 2
        axes[2].imshow(eo_real_display, cmap='RdYlGn')
    
    axes[2].set_title('Real EO')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def test_model(config, config_name, checkpoint_epoch='final'):
    """Test trained CycleGAN model"""
    
    print(f"Testing CycleGAN for {config_name} configuration...")
    config['current_config'] = config_name
    
    # Create test dataset
    band_config = config['band_configs'][config_name]
    target_bands = band_config['bands']
    
    sar_dir = config['data']['sar_dir']
    eo_dir = config['data']['eo_dir']
    
    dataset = Sen12MSDataset(sar_dir, eo_dir, target_bands)
    
    # Use same split as training (20% for testing)
    # dataset_size = len(dataset)
    # train_size = int(0.8 * dataset_size)
    # test_size = dataset_size - train_size
    # Split dataset (80-20 split)
    dataset_size = len(dataset)
    # train_size = int(0.8 * dataset_size)
    train_size = 500
    # test_size = dataset_size - train_size
    test_size = 50
    
    _, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_loader = create_data_loader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = CycleGANModel(config)
    
    # Load trained weights
    checkpoint_dir = f"checkpoints/{config_name}"
    model.load_networks(checkpoint_epoch, checkpoint_dir)
    model.eval_mode()
    
    # Create results directory
    results_dir = f"results/{config_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate model
    print("Evaluating model performance...")
    metrics = evaluate_model(model, test_loader, config_name)
    print_metrics(metrics, config_name)
    
    # Save metrics to file
    metrics_file = os.path.join(results_dir, 'evaluation_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Evaluation Results for {config_name.upper()}\n")
        f.write("=" * 50 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    # Generate sample visualizations
    print("Generating sample visualizations...")
    num_samples = min(10, len(test_dataset))
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Generating samples")):
            if i >= num_samples:
                break
                
            model.set_input(batch)
            model.forward()
            
            # Get results
            sar_input = batch['sar'][0]  # First sample in batch
            generated_eo = model.fake_B[0]  # Generated EO
            real_eo = batch['eo'][0]  # Real EO
            
            # Save visualization
            save_path = os.path.join(results_dir, f'sample_{i+1:03d}.png')
            visualize_results(sar_input, generated_eo, real_eo, save_path, config_name)
    
    print(f"Results saved to {results_dir}")
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Test CycleGAN for SAR-to-EO translation')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--band_config', type=str, choices=['rgb', 'nir_swir', 'rgb_nir'], 
                       default='rgb', help='Band configuration to test')
    parser.add_argument('--checkpoint', type=str, default='final', help='Checkpoint epoch to load')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Test model
    metrics = test_model(config, args.band_config, args.checkpoint)
    
    print("Testing completed!")

if __name__ == '__main__':
    main()
