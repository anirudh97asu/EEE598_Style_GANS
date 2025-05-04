import os
import argparse
import torch

# Import custom modules
from train import train_stylegan
from evaluate import evaluate_stylegan
from inference import run_inference

def main():
    parser = argparse.ArgumentParser(description="StyleGAN Implementation")
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Train mode
    train_parser = subparsers.add_parser("train", help="Train StyleGAN model")
    # Data and output directories
    train_parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training images")
    train_parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    
    # Training parameters
    train_parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--latent_dim", type=int, default=512, help="Latent space dimension")
    train_parser.add_argument("--learning_rate", type=float, default=0.002, help="Learning rate")
    train_parser.add_argument("--beta1", type=float, default=0.0, help="Beta1 for Adam optimizer")
    train_parser.add_argument("--beta2", type=float, default=0.99, help="Beta2 for Adam optimizer")
    
    # Regularization parameters
    train_parser.add_argument("--r1_gamma", type=float, default=10.0, help="R1 regularization weight")
    train_parser.add_argument("--path_reg_weight", type=float, default=2.0, help="Path length regularization weight")
    train_parser.add_argument("--path_batch_shrink", type=int, default=2, help="Shrink factor for path length regularization batch")
    train_parser.add_argument("--d_reg_every", type=int, default=16, help="Apply R1 regularization every N iterations")
    train_parser.add_argument("--g_reg_every", type=int, default=4, help="Apply path length regularization every N iterations")
    train_parser.add_argument("--mixing_prob", type=float, default=0.9, help="Style mixing probability")
    
    # Resolution settings
    train_parser.add_argument("--start_resolution", type=int, default=4, help="Starting resolution for progressive growing")
    train_parser.add_argument("--target_resolution", type=int, default=256, help="Target resolution")
    train_parser.add_argument("--progressive_growing", action="store_true", help="Enable progressive growing")
    
    # Logging and saving parameters
    train_parser.add_argument("--checkpoint_freq", type=int, default=5000, help="Save checkpoint every N iterations")
    train_parser.add_argument("--sample_freq", type=int, default=1000, help="Generate samples every N iterations")
    train_parser.add_argument("--log_freq", type=int, default=100, help="Log metrics every N iterations")
    train_parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    train_parser.add_argument("--use_tensorboard", action="store_true", help="Enable TensorBoard logging")
    
    # Resume training
    train_parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    
    # Device
    train_parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")
    
    # Evaluate mode
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate StyleGAN model")
    # Model and data paths
    eval_parser.add_argument("--generator_path", type=str, required=True, help="Path to generator checkpoint")
    eval_parser.add_argument("--data_dir", type=str, default=None, help="Directory with real images for FID comparison")
    eval_parser.add_argument("--output_dir", type=str, default="evaluation", help="Directory to save evaluation results")
    
    # Evaluation parameters
    eval_parser.add_argument("--n_samples", type=int, default=50000, help="Number of samples to generate")
    eval_parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation and feature extraction")
    eval_parser.add_argument("--latent_dim", type=int, default=512, help="Latent space dimension")
    eval_parser.add_argument("--truncation_psi", type=float, default=0.7, help="Truncation parameter")
    eval_parser.add_argument("--fid_real_stats", type=str, default=None, help="Path to precomputed real stats for FID")
    
    # Device
    eval_parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")
    
    # Inference mode
    infer_parser = subparsers.add_parser("infer", help="Run inference with StyleGAN model")
    # Model path
    infer_parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to generator checkpoint")
    infer_parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    
    # Inference parameters
    infer_parser.add_argument("--n_samples", type=int, default=16, help="Number of samples to generate")
    infer_parser.add_argument("--latent_dim", type=int, default=512, help="Latent space dimension")
    infer_parser.add_argument("--truncation_psi", type=float, default=0.7, help="Truncation parameter")
    infer_parser.add_argument("--generate_interpolations", action="store_true", help="Generate latent space interpolations")
    infer_parser.add_argument("--n_interpolation_steps", type=int, default=10, help="Number of steps for each interpolation")
    infer_parser.add_argument("--generate_style_mixing", action="store_true", help="Generate style mixing examples")
    infer_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    # Device
    infer_parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Set default device if not specified
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Run appropriate function based on mode
    if args.mode == "train":
        train_stylegan(**vars(args))
    elif args.mode == "evaluate":
        evaluate_stylegan(**vars(args))
    elif args.mode == "infer":
        run_inference(**vars(args))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()