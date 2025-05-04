import torch
from torch import nn, optim
import os
import argparse
from tqdm import tqdm
from math import log2
import torch.nn.functional as F

from model import Generator, Discriminator
from data import get_loader
from utils import gradient_penalty, save_checkpoint, load_checkpoint, generate_examples

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train StyleGAN")
    parser.add_argument("--dataset", type=str, default="path/to/dataset", help="Path to dataset")
    parser.add_argument("--out_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--load_model", action="store_true", help="Load model from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--start_train_img_size", type=int, default=4, help="Starting image size")
    parser.add_argument("--target_img_size", type=int, default=128, help="Target image size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs per resolution")
    parser.add_argument("--batch_sizes", type=str, default="1024,1024,1024,256,128,64", 
                        help="Batch sizes for each resolution")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--z_dim", type=int, default=256, help="Latent dimension")
    parser.add_argument("--w_dim", type=int, default=256, help="Style dimension")
    parser.add_argument("--in_channels", type=int, default=256, help="Initial channels")
    parser.add_argument("--lambda_gp", type=float, default=10, help="Gradient penalty lambda")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    
    return parser.parse_args()

def train_fn(critic, gen, loader, dataset, step, num_epochs, alpha, opt_critic, opt_gen, device, lambda_gp):
    """
    Training function for one epoch
    
    Args:
        critic: Discriminator model
        gen: Generator model
        loader: DataLoader
        dataset: Dataset object
        step: Current resolution step
        alpha: Current alpha for progressive growing
        opt_critic: Optimizer for critic
        opt_gen: Optimizer for generator
        device: Device to train on
        lambda_gp: Lambda for gradient penalty
        
    Returns:
        alpha: Updated alpha value
    """
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        noise = torch.randn(cur_batch_size, gen.z_dim).to(device)
        
        # Generate fake images
        fake = gen(noise, alpha, step)
        
        # Calculate critic scores
        critic_real = critic(real, alpha, step)
        critic_fake = critic(fake.detach(), alpha, step)
        
        # Calculate gradient penalty
        gp = gradient_penalty(critic, real, fake, alpha, step, device)
        
        # Critic loss
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))
            + lambda_gp * gp
            + (0.001) * torch.mean(critic_real ** 2)  # R1 regularization
        )
        
        # Update critic
        opt_critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()
        
        # Generator loss and update
        gen_fake = critic(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        # Update alpha for progressive growing
        alpha += cur_batch_size / (
            num_epochs * 0.5 * len(dataset)
        )
        alpha = min(alpha, 1)
        
        # Update progress bar
        loop.set_postfix(
            loss_generator=loss_gen.item(),
            gp=gp.item(),
            loss_critic=loss_critic.item(),
            alpha=alpha
        )
        
        # Monitor GPU memory every 50 batches
        if batch_idx % 50 == 0 and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                print(f"GPU {i} memory: {allocated:.2f} GB")
    
    return alpha

def main():
    args = parse_arguments()
    
    # Set up directories
    model_dir = os.path.join(args.out_dir, "model_checkpoints")
    sample_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Parse batch sizes
    batch_sizes = [int(size) for size in args.batch_sizes.split(',')]
    
    # Define factors for progressive growing
    factors = [1, 1, 1, 1, 1/2, 1/4]
    
    # Initialize models
    gen = Generator(
        z_dim=args.z_dim,
        w_dim=args.w_dim,
        in_channels=args.in_channels,
        img_channels=3,
        factors=factors
    ).to(device)
    
    critic = Discriminator(
        in_channels=args.in_channels,
        img_channels=3,
        factors=factors
    ).to(device)
    
    # Wrap models in DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        gen = torch.nn.DataParallel(gen)
        critic = torch.nn.DataParallel(critic)
    
    # Define optimizers
    opt_gen = optim.Adam([
        {'params': [param for name, param in gen.named_parameters() if 'map' not in name]},
        {'params': gen.module.map.parameters() if isinstance(gen, torch.nn.DataParallel) else gen.map.parameters(), 'lr': 1e-5}
    ], lr=args.lr, betas=(0.0, 0.99))
    
    opt_critic = optim.Adam(
        critic.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
    
    # Load checkpoint if specified
    if args.load_model and args.checkpoint:
        step, alpha = load_checkpoint(
            args.checkpoint, gen, critic, opt_gen, opt_critic, device
        )
    else:
        step = int(log2(args.start_train_img_size / 4))
        alpha = 1e-3
    
    # Number of epochs for each resolution
    num_epochs = args.num_epochs
    
    # Maximum step based on target image size
    max_step = int(log2(args.target_img_size / 4))
    
    gen.train()
    critic.train()
    
    # Training loop
    while step <= max_step:
        current_res = 4 * 2 ** step
        print(f"Training at resolution: {current_res}x{current_res}")
        
        train_loader, train_dataset = get_loader(
            current_res, batch_sizes, 3, args.dataset
        )
        
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            
            alpha = train_fn(
                critic, gen, train_loader, train_dataset, step, num_epochs, alpha, 
                opt_critic, opt_gen, device, args.lambda_gp
            )
            
            # Generate sample images
            if epoch % 2 == 0:
                generate_examples(
                    gen, step, epoch, current_res, 
                    device, sample_dir, n=16
                )
        
        print(f"Completed training for resolution: {current_res}x{current_res}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            model_dir, f"stylegan_checkpoint_res{current_res}.pth"
        )
        save_checkpoint(gen, critic, opt_gen, opt_critic, step, alpha, checkpoint_path)
        
        # Move to next resolution
        step += 1
        alpha = 1e-3

if __name__ == "__main__":
    main()