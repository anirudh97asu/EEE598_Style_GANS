import torch
import torch.nn as nn
import os
import numpy as np
from torchvision.utils import save_image

def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    """
    Compute gradient penalty for WGAN-GP
    
    Args:
        critic: Discriminator model
        real: Batch of real images
        fake: Batch of generated images
        alpha: Current alpha for progressive growing
        train_step: Current step/resolution level
        device: Device to compute on
        
    Returns:
        Gradient penalty loss term
    """
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)
 
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    
    return gradient_penalty

def save_checkpoint(gen, critic, opt_gen, opt_critic, step, alpha, filename):
    """
    Save model checkpoint
    
    Args:
        gen: Generator model
        critic: Discriminator model
        opt_gen: Generator optimizer
        opt_critic: Discriminator optimizer
        step: Current step/resolution level
        alpha: Current alpha for progressive growing
        filename: Path to save checkpoint
    """
    checkpoint = {
        'generator': gen.module.state_dict() if isinstance(gen, torch.nn.DataParallel) else gen.state_dict(),
        'critic': critic.module.state_dict() if isinstance(critic, torch.nn.DataParallel) else critic.state_dict(),
        'opt_gen': opt_gen.state_dict(),
        'opt_critic': opt_critic.state_dict(),
        'step': step,
        'alpha': alpha
    }
    
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(path, gen, critic, opt_gen, opt_critic, device):
    """
    Load model checkpoint
    
    Args:
        path: Path to checkpoint file
        gen: Generator model
        critic: Discriminator model
        opt_gen: Generator optimizer
        opt_critic: Discriminator optimizer
        device: Device to load model on
        
    Returns:
        step: Current step/resolution level
        alpha: Current alpha for progressive growing
    """
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)
    
    # Handle loading state dict for DataParallel models
    if isinstance(gen, torch.nn.DataParallel):
        gen.module.load_state_dict(checkpoint['generator'])
        critic.module.load_state_dict(checkpoint['critic'])
    else:
        gen.load_state_dict(checkpoint['generator'])
        critic.load_state_dict(checkpoint['critic'])
    
    opt_gen.load_state_dict(checkpoint['opt_gen'])
    opt_critic.load_state_dict(checkpoint['opt_critic'])
    
    # Get training state
    step = checkpoint.get('step', 0)
    alpha = checkpoint.get('alpha', 1e-3)
    
    print(f"Checkpoint loaded. Current step: {step}, alpha: {alpha}")
    
    return step, alpha

def generate_examples(gen, steps, epoch, img_size, device, output_dir, n=16):
    """
    Generate and save sample images
    
    Args:
        gen: Generator model
        steps: Current step/resolution level
        epoch: Current epoch
        img_size: Current image size
        device: Device to generate on
        output_dir: Directory to save samples
        n: Number of samples to generate
    """
    save_dir = os.path.join(output_dir, f"size_{img_size}")
    os.makedirs(save_dir, exist_ok=True)
    
    gen.eval()
    alpha = 1.0
    
    # Generate a grid of images
    with torch.no_grad():
        noise = torch.randn(n, gen.module.z_dim if isinstance(gen, torch.nn.DataParallel) else gen.z_dim).to(device)
        fake = gen(noise, alpha, steps)
        
        # Save individual images
        for i in range(min(n, fake.shape[0])):
            save_image(
                fake[i:i+1] * 0.5 + 0.5,  # Denormalize
                f"{save_dir}/epoch_{epoch}_img_{i}.png"
            )
        
        # Save a grid of images
        save_image(
            fake * 0.5 + 0.5,  # Denormalize
            f"{save_dir}/epoch_{epoch}_grid.png",
            nrow=int(np.sqrt(n))
        )
    
    gen.train()
    print(f"Generated {n} samples at resolution {img_size}x{img_size}")

def visualize_model_outputs(gen, critic, dataloader, steps, alpha, device):
    """
    Visualize model outputs for debugging
    
    Args:
        gen: Generator model
        critic: Discriminator model
        dataloader: DataLoader for real images
        steps: Current step/resolution level
        alpha: Current alpha for progressive growing
        device: Device to generate on
    """
    import matplotlib.pyplot as plt
    
    # Get a batch of real images
    real_images, _ = next(iter(dataloader))
    real_images = real_images.to(device)
    
    # Generate fake images
    with torch.no_grad():
        z = torch.randn(real_images.shape[0], gen.module.z_dim if isinstance(gen, torch.nn.DataParallel) else gen.z_dim).to(device)
        fake_images = gen(z, alpha, steps)
    
    # Display comparison
    plt.figure(figsize=(12, 8))
    
    # Display real images
    for i in range(min(4, real_images.shape[0])):
        plt.subplot(2, 4, i + 1)
        plt.imshow(real_images[i].cpu().permute(1, 2, 0) * 0.5 + 0.5)
        plt.title("Real")
        plt.axis("off")
    
    # Display fake images
    for i in range(min(4, fake_images.shape[0])):
        plt.subplot(2, 4, i + 5)
        plt.imshow(fake_images[i].cpu().permute(1, 2, 0) * 0.5 + 0.5)
        plt.title("Fake")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and display critic scores
    with torch.no_grad():
        real_scores = critic(real_images, alpha, steps)
        fake_scores = critic(fake_images, alpha, steps)
    
    print(f"Average real score: {real_scores.mean().item():.4f}")
    print(f"Average fake score: {fake_scores.mean().item():.4f}")
    print(f"Gap: {(real_scores.mean() - fake_scores.mean()).item():.4f}")


def generate_latent_walk(gen, steps, n_samples=10, n_interpolation=10, device="cpu"):
    """
    Generate a latent space walk between two points
    
    Args:
        gen: Generator model
        steps: Current step/resolution level
        n_samples: Number of sample pairs
        n_interpolation: Number of interpolation steps between pairs
        device: Device to generate on
        
    Returns:
        List of images showing latent space interpolation
    """
    gen.eval()
    alpha = 1.0
    z_dim = gen.module.z_dim if isinstance(gen, torch.nn.DataParallel) else gen.z_dim
    
    # Generate random pairs of latent vectors
    z_anchors = torch.randn(n_samples + 1, z_dim).to(device)
    
    # Interpolate between pairs
    interpolated_images = []
    
    with torch.no_grad():
        for i in range(n_samples):
            # Linear interpolation between two z vectors
            for t in range(n_interpolation + 1):
                # Interpolation factor (0 to 1)
                factor = t / n_interpolation
                
                # Interpolate in the latent space
                z_interp = z_anchors[i] * (1 - factor) + z_anchors[i + 1] * factor
                z_interp = z_interp.unsqueeze(0)  # Add batch dimension
                
                # Generate image from interpolated latent vector
                img = gen(z_interp, alpha, steps)
                img = img[0] * 0.5 + 0.5  # Denormalize
                interpolated_images.append(img.cpu())
    
    gen.train()
    return interpolated_images


def visualize_latent_walk(images, rows=None, cols=None, figsize=(15, 8)):
    """
    Visualize a latent space walk
    
    Args:
        images: List of images from latent walk
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    from math import ceil, sqrt
    
    n_images = len(images)
    
    # Set default grid dimensions if not provided
    if rows is None and cols is None:
        cols = ceil(sqrt(n_images))
        rows = ceil(n_images / cols)
    elif rows is None:
        rows = ceil(n_images / cols)
    elif cols is None:
        cols = ceil(n_images / rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, img in enumerate(images):
        if i < n_images:
            axes[i].imshow(img.permute(1, 2, 0).numpy())
            axes[i].set_title(f"Step {i}")
            axes[i].axis("off")
    
    # Hide any empty subplots
    for i in range(n_images, rows * cols):
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()


def calculate_fid(real_images, fake_images):
    """
    Calculate the Fréchet Inception Distance (FID) score
    
    Args:
        real_images: Batch of real images
        fake_images: Batch of generated images
        
    Returns:
        FID score (lower is better)
    """
    # This is a simplified version - a real implementation would use the InceptionV3 model
    # and calculate statistics on activations
    
    try:
        from scipy import linalg
        import numpy as np
        from torchvision.models import inception_v3
        import torch.nn.functional as F
    except ImportError:
        print("FID calculation requires scipy, numpy, and torchvision")
        return None
    
    def get_activations(images, model):
        model.eval()
        with torch.no_grad():
            pred = model(images)
        return pred.detach().cpu().numpy()
    
    # Preprocess images for Inception model
    def preprocess(imgs, target_size=299):
        imgs = F.interpolate(imgs, size=(target_size, target_size), mode='bilinear', align_corners=False)
        return imgs
    
    # Get InceptionV3 model (simplified, actual implementation would be more complex)
    inception_model = inception_v3(pretrained=True).to(real_images.device)
    inception_model.eval()
    
    # Get activations
    real_activations = get_activations(preprocess(real_images), inception_model)
    fake_activations = get_activations(preprocess(fake_images), inception_model)
    
    # Calculate mean and covariance
    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    
    mu_fake = np.mean(fake_activations, axis=0)
    sigma_fake = np.cov(fake_activations, rowvar=False)
    
    # Calculate FID
    ssdiff = np.sum((mu_real - mu_fake) ** 2.0)
    covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return fid


def calculate_gpu_memory_usage():
    """
    Calculate and print GPU memory usage
    
    Returns:
        Dictionary of memory usage per GPU
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return {}
    
    memory_usage = {}
    
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        reserved_memory = torch.cuda.memory_reserved(i) / 1e9
        allocated_memory = torch.cuda.memory_allocated(i) / 1e9
        free_memory = total_memory - reserved_memory
        
        memory_usage[f"gpu_{i}"] = {
            "total_memory_GB": total_memory,
            "reserved_memory_GB": reserved_memory,
            "allocated_memory_GB": allocated_memory,
            "free_memory_GB": free_memory
        }
        
        print(f"GPU {i} - Total: {total_memory:.2f} GB | Reserved: {reserved_memory:.2f} GB | Allocated: {allocated_memory:.2f} GB | Free: {free_memory:.2f} GB")
    
    return memory_usage