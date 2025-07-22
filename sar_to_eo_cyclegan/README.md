<<<<<<< HEAD
🎯 Project Overview
This project implements a CycleGAN model to perform unpaired image-to-image translation from SAR images (Sentinel-1) to EO images (Sentinel-2). The model learns to generate optical-like images from radar data, enabling all-weather Earth observation capabilities.

Key Objectives
Generate realistic EO images from SAR input using deep learning

Train multiple configurations for different spectral band combinations

Evaluate translation quality using comprehensive metrics

Enable all-weather monitoring by bridging SAR and EO modalities

Why CycleGAN?
No paired data required - works with temporally misaligned SAR/EO images

Bidirectional translation - learns both SAR→EO and EO→SAR mappings

Cycle consistency - preserves structural and semantic information

Proven architecture - established success in image-to-image translation

✨ Features
Three Band Configurations:

RGB (B4, B3, B2) - Natural color representation

NIR/SWIR (B8, B11, B5) - Vegetation and land analysis

RGB+NIR (B4, B3, B2, B8) - Enhanced analytical capability

Comprehensive Evaluation:

PSNR, SSIM, MAE, RMSE metrics

NDVI correlation analysis

Visual comparison outputs

Production-Ready Code:

Modular architecture

Configuration-driven training

Tensorboard integration

Automatic checkpointing

📊 Dataset
Sen12MS Dataset
Source: Technical University of Munich

Content: 180,662 triplets of SAR/EO/Land cover patches

Resolution: 256×256 pixels, 10m ground sampling distance

Temporal Coverage: Full year 2017 data

License: CC-BY open access

Winter Season Data (This Project)
SAR: ROIs2017_winter_s1.tar.gz (~2.1 GB)

EO: ROIs2017_winter_s2.tar.gz (~8.4 GB)

Time Period: December 2016 - February 2017

Download: https://dataserv.ub.tum.de/s/m1474000

Data Specifications
SAR Channels: VV, VH polarizations in dB scale

EO Bands: 13 Sentinel-2 spectral bands

Format: GeoTIFF files

Preprocessing: Radiometric calibration, geometric correction applied

🚀 Installation
Prerequisites
Python 3.8+

CUDA-capable GPU (recommended)

16GB+ RAM

50GB+ free disk space

Step 1: Clone and Setup Environment
bash
# Clone the project
git clone <repository-url>
cd sar_to_eo_cyclegan

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Step 2: Download Dataset
bash
# Create data directories
mkdir -p data/raw

# Download Sen12MS winter data
cd data/raw
wget https://dataserv.ub.tum.de/s/m1474000/download?path=%2F&files=ROIs2017_winter_s1.tar.gz
wget https://dataserv.ub.tum.de/s/m1474000/download?path=%2F&files=ROIs2017_winter_s2.tar.gz

# Extract archives
tar -xzf ROIs2017_winter_s1.tar.gz
tar -xzf ROIs2017_winter_s2.tar.gz

cd ../..
Step 3: Verify Installation
bash
# Test dataset loading
python -c "from utils.dataset import Sen12MSDataset; print('Installation successful!')"
📁 Project Structure
text
sar_to_eo_cyclegan/
├── 📁 data/
│   ├── 📁 raw/                     # Downloaded dataset
│   │   ├── ROIs2017_winter_s1/     # SAR data
│   │   └── ROIs2017_winter_s2/     # EO data
│   └── 📁 processed/               # Preprocessed data cache
├── 📁 models/                      # Neural network architectures
│   ├── __init__.py
│   ├── 🧠 cyclegan.py             # Main CycleGAN model
│   ├── 🏗️ networks.py             # Generator/Discriminator architectures
│   └── 📊 losses.py               # Loss functions (GAN, Cycle, Identity)
├── 📁 utils/                       # Utility functions
│   ├── __init__.py
│   ├── 🗃️ dataset.py              # Dataset loading and preprocessing
│   ├── ⚙️ preprocessing.py        # Image preprocessing functions
│   └── 📈 metrics.py              # Evaluation metrics
├── 📁 configs/
│   └── ⚙️ config.yaml             # Training/model configuration
├── 📁 checkpoints/                 # Saved model weights
│   ├── rgb/
│   ├── nir_swir/
│   └── rgb_nir/
├── 📁 logs/                        # Tensorboard logs
├── 📁 results/                     # Test results and visualizations
├── 📁 notebooks/
│   └── 📊 visualization.ipynb     # Advanced result analysis
├── 🎯 main.py                     # Main entry point
├── 🏋️ train.py                    # Training script
├── 🧪 test.py                     # Testing/evaluation script
├── 📋 requirements.txt            # Python dependencies
└── 📖 README.md                   # This file
Key File Descriptions
File	Purpose	What it does
main.py	Entry point	Orchestrates training/testing for all configurations
train.py	Training logic	Implements the complete training loop with logging
test.py	Evaluation	Loads trained models and computes performance metrics
models/cyclegan.py	CycleGAN wrapper	Combines generators, discriminators, and loss functions
models/networks.py	Neural architectures	ResNet generators and PatchGAN discriminators
utils/dataset.py	Data handling	Custom PyTorch dataset for Sen12MS data
utils/preprocessing.py	Data preprocessing	SAR/EO image preprocessing and normalization
configs/config.yaml	Configuration	All hyperparameters and settings in one place
🎮 Usage
Quick Start (All Configurations)
bash
# Train and test all three band configurations
python main.py --mode both --band_config all
Individual Configuration Training
bash
# Train only RGB configuration
python main.py --mode train --band_config rgb

# Train NIR/SWIR configuration  
python main.py --mode train --band_config nir_swir

# Train RGB+NIR configuration
python main.py --mode train --band_config rgb_nir
Testing Only
bash
# Test specific configuration with trained model
python test.py --band_config rgb --checkpoint final

# Test with specific epoch checkpoint
python test.py --band_config rgb --checkpoint 150
Advanced Usage
bash
# Custom configuration file
python main.py --config custom_config.yaml --band_config rgb

# Training with specific parameters
python train.py --config configs/config.yaml --band_config rgb
Monitoring Training
bash
# Start Tensorboard (in separate terminal)
tensorboard --logdir logs/

# View at http://localhost:6006
⚙️ Configuration
Band Configurations
Configuration	Bands	Channels	Use Case
RGB	B4, B3, B2	3	Natural color visualization
NIR/SWIR	B8, B11, B5	3	Vegetation/land cover analysis
RGB+NIR	B4, B3, B2, B8	4	Enhanced multispectral analysis
Key Hyperparameters
text
# Training Configuration
training:
  batch_size: 4          # Adjust based on GPU memory
  num_epochs: 200        # Standard for CycleGAN
  lr: 0.0002            # Initial learning rate
  lambda_cycle: 10.0     # Cycle consistency weight
  lambda_identity: 5.0   # Identity loss weight

# Model Architecture
model:
  input_nc: 2           # SAR channels (VV, VH)
  ngf: 64              # Generator filters
  ndf: 64              # Discriminator filters
  norm: "instance"      # Normalization type
Customization Options
Modify training duration:

text
training:
  num_epochs: 100  # Faster training
  save_freq: 5     # More frequent checkpoints
Adjust model complexity:

text
model:
  ngf: 32          # Lighter generator
  n_blocks: 6      # Fewer residual blocks
Change loss weights:

text
training:
  lambda_cycle: 15.0   # Stronger cycle consistency
  lambda_identity: 0   # Disable identity loss
📊 Results
Expected Performance (Typical Results)
Configuration	PSNR (dB)	SSIM	MAE	RMSE	NDVI Corr
RGB	20-25	0.4-0.6	0.15-0.25	0.20-0.30	0.6-0.8
NIR/SWIR	18-23	0.3-0.5	0.18-0.28	0.22-0.32	-
RGB+NIR	19-24	0.4-0.6	0.16-0.26	0.21-0.31	0.7-0.9
Output Files
During Training:

checkpoints/{config}/net_G_A_epoch_{N}.pth - Generator weights

logs/{config}/ - Tensorboard training logs

Console output with loss progression

After Testing:

results/{config}/evaluation_metrics.txt - Numerical results

results/{config}/sample_XXX.png - Visual comparisons

Side-by-side SAR input, generated EO, real EO

Interpreting Results
Good Results Indicators:

✅ PSNR > 20 dB

✅ SSIM > 0.4

✅ Generated images preserve spatial structures

✅ NDVI correlation > 0.6 (for vegetation-sensitive configs)

Training Success Signs:

Generator loss decreases over epochs

Discriminator loss stabilizes around 0.5

Cycle loss decreases significantly

Generated images become more realistic over time

📈 Evaluation Metrics
Pixel-Level Metrics
Peak Signal-to-Noise Ratio (PSNR)

Measures reconstruction quality

Higher is better (typically 18-27 dB)

Formula: 10 * log10(MAX²/MSE)

Structural Similarity Index (SSIM)

Evaluates structural preservation

Range: , higher is better

Considers luminance, contrast, structure

Mean Absolute Error (MAE)

Average pixel-wise difference

Lower is better

Formula: mean(|generated - real|)

Application-Specific Metrics
NDVI Correlation

Vegetation index consistency

Important for land cover applications

Computed when NIR and Red bands available

Formula: corr(NDVI_generated, NDVI_real)

Visual Quality Assessment
Generated Image Quality:

Spatial structure preservation

Texture realism

Absence of artifacts

Color/intensity consistency

🔧 Troubleshooting
Common Issues and Solutions
Problem: CUDA out of memory

text
# Solution: Reduce batch size in config.yaml
training:
  batch_size: 2  # or even 1
Problem: Dataset loading errors

bash
# Solution: Verify data directory structure
ls data/raw/ROIs2017_winter_s1/  # Should contain subdirectories
ls data/raw/ROIs2017_winter_s2/  # Should contain subdirectories
Problem: Training loss not decreasing

Check learning rate (try 0.0001)

Verify data normalization

Ensure cycle loss weight is appropriate (10.0)

Monitor discriminator/generator balance

Problem: Poor visual results

Train longer (200+ epochs)

Adjust cycle loss weight

Check if identity loss helps

Verify input data quality

Problem: Import errors

bash
# Solution: Ensure proper Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Or use relative imports
Performance Optimization
Faster Training:

Use mixed precision training

Reduce image resolution (modify dataset)

Use fewer residual blocks (n_blocks: 6)

Increase batch size if GPU memory allows

Better Results:

Train longer (300+ epochs)

Use learning rate scheduling

Experiment with loss weights

Try different normalization schemes

Memory Management
Large Dataset Handling:

python
# In dataset.py, add data caching
class Sen12MSDataset(Dataset):
    def __init__(self, ..., cache_size=1000):
        self.cache = {}
        self.cache_size = cache_size
GPU Memory Optimization:

python
# Clear cache periodically
if epoch % 10 == 0:
    torch.cuda.empty_cache()
🤝 Contributing
Development Setup
bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black .
isort .
Adding New Features
New Band Configuration:

Add configuration to configs/config.yaml

Update band indexing in utils/dataset.py

Modify metrics calculation if needed

New Loss Function:

Implement in models/losses.py

Integrate into models/cyclegan.py

Add hyperparameter to config

New Metrics:

Add function to utils/metrics.py

Update test.py evaluation loop

Modify result visualization

Code Style Guidelines
Follow PEP 8 style guidelines

Use type hints where appropriate

Add docstrings for all functions

Keep functions focused and modular

📚 References
Academic Papers
Zhu, J.Y., et al. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." ICCV 2017.

Schmitt, M., et al. "SEN12MS – A Curated Dataset of Georeferenced Multi-Spectral Sentinel-1/2 Imagery for Deep Learning and Data Fusion." ISPRS Annals 2019.

Isola, P., et al. "Image-to-Image Translation with Conditional Adversarial Networks." CVPR 2017.

Technical Resources
Sentinel-1 Product Specification

Sentinel-2 Product Specification

PyTorch CycleGAN Implementation

Dataset Information
Sen12MS Dataset Paper

Technical University of Munich Data Portal

ESA Copernicus Data Access

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🏷️ Citation
If you use this code in your research, please cite:

text
@misc{sar_to_eo_cyclegan,
  title={SAR-to-EO Image Translation Using CycleGAN},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-username/sar-to-eo-cyclegan}}
}
📞 Support
For questions and support:

📧 Email: your.email@example.com

🐛 Issues: GitHub Issues

💬 Discussions: GitHub Discussions

Happy translating! 🛰️ → 🌍
=======
# SAR_TO_EO_CYCLEGAN
CONVERTS IMAGES FROM SAR TO EO
>>>>>>> bdd3ab79df270fd45ff2b476ab13d3006447a30d
