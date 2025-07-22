import os
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

from utils.dataset import Sen12MSDataset, create_data_loader, validate_dataset
from utils.preprocessing import save_normalization_params
from models.cyclegan import CycleGANModel

def train_model(config, config_name):
    """Train CycleGAN model for specific configuration"""
    
    print(f"Training CycleGAN for {config_name} configuration...")
    config['current_config'] = config_name
    
    # Create datasets
    band_config = config['band_configs'][config_name]
    target_bands = band_config['bands']

    # Dataset paths
    sar_dir = config['data']['sar_dir']
    eo_dir = config['data']['eo_dir']

    # Create dataset
    dataset = Sen12MSDataset(sar_dir, eo_dir, target_bands)
    # validate_dataset(dataset, num_samples=3)

    # Split dataset (80-20 split)
    dataset_size = len(dataset)
    # train_size = int(0.8 * dataset_size)
    train_size = 500
    # test_size = dataset_size - train_size
    test_size = 50


    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = create_data_loader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        shuffle=True
    )

    test_loader = create_data_loader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = CycleGANModel(config)
    
    # Create directories
    checkpoint_dir = f"checkpoints/{config_name}"
    log_dir = f"logs/{config_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    save_freq = config['training']['save_freq']
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train_mode()
        
        # Training phase
        train_losses = {'D_A': [], 'D_B': [], 'G_A': [], 'G_B': [], 'cycle': [], 'idt_A': [], 'idt_B': []}
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for i, batch in enumerate(pbar):
            model.set_input(batch)
            model.optimize_parameters()
            
            # Get current losses
            losses = model.get_current_losses()
            for key, value in losses.items():
                train_losses[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f"{losses['G_A'] + losses['G_B']:.4f}",
                'D_loss': f"{losses['D_A'] + losses['D_B']:.4f}",
                'Cycle': f"{losses['cycle']:.4f}"
            })
        
        # Calculate mean losses
        mean_losses = {key: sum(values) / len(values) for key, values in train_losses.items()}
        
        # Log to tensorboard
        for key, value in mean_losses.items():
            writer.add_scalar(f'Loss/{key}', value, epoch)
        
        writer.add_scalar('Loss/G_total', mean_losses['G_A'] + mean_losses['G_B'], epoch)
        writer.add_scalar('Loss/D_total', mean_losses['D_A'] + mean_losses['D_B'], epoch)
        
        # Update learning rate
        model.update_learning_rate()
        
        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}]:")
        print(f"  G_loss: {mean_losses['G_A'] + mean_losses['G_B']:.4f}")
        print(f"  D_loss: {mean_losses['D_A'] + mean_losses['D_B']:.4f}")
        print(f"  Cycle_loss: {mean_losses['cycle']:.4f}")
        
        # Save checkpoints
        if (epoch + 1) % save_freq == 0:
            model.save_networks(epoch + 1, checkpoint_dir)
            
            # Save sample results
            model.eval_mode()
            with torch.no_grad():
                test_batch = next(iter(test_loader))
                model.set_input(test_batch)
                model.forward()
                
                # Save sample images (implement visualization here if needed)
                
        # Save final model
        if epoch == num_epochs - 1:
            model.save_networks('final', checkpoint_dir)
    
    writer.close()
    print(f"Training completed for {config_name}")
    
    return model, test_loader

def main():
    parser = argparse.ArgumentParser(description='Train CycleGAN for SAR-to-EO translation')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--band_config', type=str, choices=['rgb', 'nir_swir', 'rgb_nir'], 
                       default='rgb', help='Band configuration to train')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train model
    model, test_loader = train_model(config, args.band_config)
    
    print("Training completed!")

if __name__ == '__main__':
    main()
