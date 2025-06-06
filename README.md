# StyleGAN for Anime Face Generation

This repository contains a PyTorch implementation of StyleGAN with progressive growing for generating high-quality anime faces. The code is designed to be modular, well-documented, and easy to use.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Generating Images](#generating-images)
  - [Latent Space Exploration](#latent-space-exploration)
- [Model Architecture](#model-architecture)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Overview

StyleGAN (Style-based Generative Adversarial Network) introduced by NVIDIA is a state-of-the-art architecture for generating high-quality images. This implementation includes progressive growing, which starts training at low resolutions and gradually increases the resolution for stable training and better results.

This repository is specifically tuned for generating anime faces using the Anime Face Dataset from Kaggle.

## Features

- **Progressive Growing**: Start training at low resolutions (4×4) and gradually increase to target resolution (up to 1024×1024)
- **Style-based Generator**: Separates high-level attributes from stochastic variation
- **Mapping Network**: Maps input latent code to an intermediate latent space for better disentanglement
- **Adaptive Instance Normalization (AdaIN)**: Controls style at each convolution layer
- **Noise Injection**: Adds stochastic detail to generated images
- **Weight Scaling**: Improves training stability
- **Multi-GPU Support**: DataParallel implementation for training on multiple GPUs
- **Checkpoint Saving/Loading**: Resume training from checkpoints
- **FID Score Calculation**: Evaluate the quality of generated images
- **Latent Space Walking**: Generate interpolations between points in the latent space

## Dataset

This implementation uses the [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) from Kaggle, which contains over 63,000 high-quality anime character faces cropped from anime images.

![image](https://github.com/user-attachments/assets/3f140278-294f-4bd7-8510-8663be2cc411)

### Dataset Details
- **Size**: ~36,000 images
- **Resolution**: 64*64
- **Format**: PNG/JPG images
- **Download Size**: ~120MB

### Dataset Characteristics
The Anime Face Dataset features a diverse collection of anime character faces with various:
- Hairstyles and colors
- Eye shapes and colors
- Facial expressions
- Art styles
- Lighting conditions

This diversity makes it an excellent dataset for training a StyleGAN model, allowing it to learn the unique characteristics of anime faces and generate high-quality, diverse anime character faces.

## Installation

### Prerequisites
- Python 3.7+
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/anirudh97asu/EEE598_Style_GANS
cd src
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) from Kaggle and extract it to a folder of your choice.

## Usage

### Training

To train the StyleGAN model with default parameters:

```bash
python train.py --dataset path/to/anime_face_dataset --out_dir output
```

#### Important Parameters:

- `--dataset`: Path to the dataset directory
- `--out_dir`: Output directory for model checkpoints and samples
- `--start_train_img_size`: Starting image size (default: 4)
- `--target_img_size`: Target image size (default: 128)
- `--num_epochs`: Number of epochs per resolution (default: 10)
- `--batch_sizes`: Batch sizes for each resolution (default: "1024,1024,1024,256,128,64")
- `--lr`: Learning rate (default: 1e-3)
- `--z_dim`: Latent dimension (default: 256)
- `--w_dim`: Style dimension (default: 256)
- `--in_channels`: Initial channels (default: 256)
- `--lambda_gp`: Gradient penalty lambda (default: 10)
- `--load_model`: Flag to load a model from checkpoint
- `--checkpoint`: Path to checkpoint file

### Notebook Execution

For an interactive experience, you can run the training process in a Jupyter notebook:

```python
import torch
from model import Generator, Discriminator
from train import train_fn
from data import get_loader
from utils import save_checkpoint, generate_examples

# Configuration
dataset_path = "path/to/anime_face_dataset"
out_dir = "output"
z_dim = 256
w_dim = 256
in_channels = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_img_size = 4
target_img_size = 128
num_epochs = 10
batch_sizes = [1024, 1024, 1024, 256, 128, 64]
lr = 1e-3
lambda_gp = 10

# Initialize models (see train.py for full implementation)
gen = Generator(z_dim, w_dim, in_channels).to(device)
critic = Discriminator(in_channels).to(device)

# Start training
# ... (refer to train.py for the full training loop)
```

### Generating Images

You can generate images using a trained model:

```python
from utils import generate_examples, load_checkpoint
from model import Generator, Discriminator
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models
gen = Generator(z_dim=256, w_dim=256, in_channels=256).to(device)
critic = Discriminator(in_channels=256).to(device)
opt_gen = torch.optim.Adam(gen.parameters(), lr=1e-3)
opt_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

# Load checkpoint
step, alpha = load_checkpoint(
    "path/to/checkpoint.pth", 
    gen, critic, opt_gen, opt_critic, device
)

# Generate examples
generate_examples(
    gen=gen,
    steps=step,
    epoch=0,
    img_size=4 * 2 ** step,
    device=device,
    output_dir="output/samples",
    n=16
)
```

### Latent Space Exploration

You can explore the latent space to see smooth transitions between different faces:

```python
from utils import generate_latent_walk, visualize_latent_walk
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load generator
gen = Generator(z_dim=256, w_dim=256, in_channels=256).to(device)
load_checkpoint("path/to/checkpoint.pth", gen, None, None, None, device)

# Generate latent walk
images = generate_latent_walk(
    gen=gen,
    steps=4,  # Resolution step
    n_samples=5,  # Number of sample pairs
    n_interpolation=10,  # Number of interpolation steps
    device=device
)

# Visualize the walk
visualize_latent_walk(images, cols=11)
```

## Model Architecture

### Generator Architecture

The generator consists of:

1. **Mapping Network**: 8-layer MLP that maps the input latent code z to an intermediate latent code w
2. **Synthesis Network**: Progressive network that generates images, starting from a learned 4×4 constant
3. **Style Modulation**: AdaIN layers that apply style at each convolution layer
4. **Noise Injection**: Random noise is added to provide stochastic variation

```
Input: Latent vector z ∈ ℝ^256
↓
Mapping Network (8 FC layers)
↓
Intermediate latent code w ∈ ℝ^256
↓
4×4 Constant → Conv → AdaIN → LeakyReLU
↓
Progressive blocks with upsampling, style modulation, and noise injection
↓
ToRGB layer
↓
Output: Generated image
```

### Discriminator Architecture

The discriminator is a progressive CNN that:

1. **FromRGB layers**: Convert RGB images to feature maps
2. **Progressive blocks**: Process images at different resolutions
3. **Minibatch Standard Deviation**: Adds a statistical feature to prevent mode collapse
4. **Final blocks**: Final convolutions and output layer

```
Input: Image
↓
FromRGB layer
↓
Progressive blocks with downsampling
↓
Minibatch Standard Deviation
↓
Final convolutions
↓
Output: Real/Fake score
```

## Implementation Details

### Progressive Growing

The training starts at a 4×4 resolution and progressively grows to the target resolution. For each resolution:

1. Train the generator and discriminator at the current resolution
2. Fade in the next resolution by linearly interpolating between upsampled outputs from the current resolution and outputs from the next resolution
3. Once the fade-in is complete, train at the new resolution

### Weight Scaling

Weight scaling is implemented to stabilize training by ensuring proper initialization:

```python
class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        # ...
```

### AdaIN Implementation

Adaptive Instance Normalization modulates feature maps based on style information:

```python
class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale = WSLinear(w_dim, channels)
        self.style_bias = WSLinear(w_dim, channels)
        
    def forward(self, x, w):
        x = self.instance_norm(x)
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale * x + style_bias
```

### WGAN-GP Loss

The implementation uses Wasserstein GAN with gradient penalty for stable training:

```python
# Critic loss
loss_critic = (
    -(torch.mean(critic_real) - torch.mean(critic_fake))
    + lambda_gp * gp
    + (0.001) * torch.mean(critic_real ** 2)  # R1 regularization
)

# Generator loss
loss_gen = -torch.mean(gen_fake)
```

## Results

### Training Progress

The model progressively generates higher resolution images as training progresses. Below is the typical progression:

1. 4×4 resolution: Basic color and shape
2. 8×8 resolution: Rough facial structure
3. 16×16 resolution: More defined facial features
4. 32×32 resolution: Clear anime face structure
5. 64×64 resolution: Detailed features, expressions
6. 128×128 resolution: High-quality anime faces

### Sample Generation Video

*[![Generation across various Resolutions](![image](https://github.com/user-attachments/assets/ceba4648-05db-48fa-8406-6217c7c82aa5)
)](https://youtu.be/l49hhokTm7k)

Click the image above to watch the tutorial video on YouTube.

### Learned Represents across various Resolutions after Training

![image](https://github.com/user-attachments/assets/bbafc49c-b625-4f75-95cd-b22a9ebfcb07)


## Troubleshooting

### Common Issues

#### Out of Memory Errors
- Reduce batch size for higher resolutions
- Reduce model size (in_channels parameter)
- Try training on a lower target resolution

```bash
python train.py --dataset path/to/anime_face_dataset --batch_sizes "512,512,256,128,64,32" --in_channels 128 --target_img_size 64
```

#### Mode Collapse
- Ensure proper gradient penalty (lambda_gp)
- Check discriminator/generator balance
- Try different learning rates

#### Slow Training
- Reduce resolution or batch size
- Enable multi-GPU training if available
- Use a more powerful GPU

### GPU Memory Monitoring

The training script monitors GPU memory usage every 50 batches:

```
GPU 0 memory: 8.45 GB
```

## References

1. Karras, T., Laine, S., & Aila, T. (2019). A Style-Based Generator Architecture for Generative Adversarial Networks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*
   
2. Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2018). Progressive Growing of GANs for Improved Quality, Stability, and Variation. In *International Conference on Learning Representations*

3. [Anime Face Dataset on Kaggle](https://www.kaggle.com/datasets/splcher/animefacedataset)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NVIDIA for the original StyleGAN paper and implementation
- Kaggle and the creators of the Anime Face Dataset
