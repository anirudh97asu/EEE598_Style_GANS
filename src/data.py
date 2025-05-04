import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from math import log2

def get_loader(image_size, batch_sizes, channels_img, dataset_path):
    """
    Create a data loader for the specified image size
    
    Args:
        image_size: Size to resize images to (square)
        batch_sizes: List of batch sizes for different resolutions
        channels_img: Number of image channels
        dataset_path: Path to the dataset directory
        
    Returns:
        loader: DataLoader for the dataset
        dataset: The dataset object
    """
    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)),
         transforms.ToTensor(),
         transforms.Normalize(
            [0.5 for _ in range(channels_img)],
            [0.5 for _ in range(channels_img)],
         )
        ]
    )
    
    batch_size = batch_sizes[int(log2(image_size/4))]
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return loader, dataset