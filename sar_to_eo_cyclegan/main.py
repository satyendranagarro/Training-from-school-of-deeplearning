import os
import yaml
import argparse
from train import train_model
from test import test_model


def main():
    parser = argparse.ArgumentParser(description='SAR-to-EO CycleGAN Project')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'],
                        default='both', help='Mode: train, test, or both')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--band_config', type=str, choices=['rgb', 'nir_swir', 'rgb_nir', 'all'],
                        default='all', help='Band configuration(s) to run')
    parser.add_argument('--checkpoint', type=str, default='final',
                        help='Checkpoint epoch to load (used in test mode)')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Determine which configurations to run
    if args.band_config == 'all':
        configs_to_run = ['rgb', 'nir_swir', 'rgb_nir']
    else:
        configs_to_run = [args.band_config]

    print("=== SAR-to-EO CycleGAN Project ===")
    print(f"Mode: {args.mode}")
    print(f"Configurations: {configs_to_run}")
    print("=" * 40)

    for config_name in configs_to_run:
        print(f"\n>>> Processing {config_name.upper()} configuration...")

        if args.mode in ['train', 'both']:
            print(f"Training {config_name}...")
            model, test_loader = train_model(config, config_name)

        if args.mode in ['test', 'both']:
            print(f"Testing {config_name}...")
            test_model(config, config_name, checkpoint_epoch=args.checkpoint)

    print("\n=== Project Completed ===")
    print("Check the following directories for results:")
    print("- checkpoints/: Trained model weights")
    print("- logs/: Training logs and tensorboard files")
    print("- results/: Test results and visualizations")


if __name__ == '__main__':
    main()
