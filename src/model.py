import torch
from torch import nn
import torch.nn.functional as F
from math import log2

class WSLinear(nn.Module):
    """Weight-scaled Linear layer for improved training stability"""
    def __init__(self, in_features, out_features):
        super(WSLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (2/in_features) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias

class PixelNorm(nn.Module):
    """Pixel-wise feature vector normalization"""
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8
        
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)

class MappingNetwork(nn.Module):
    """The mapping network that transforms latent z to intermediate latent w"""
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            PixelNorm(),
            WSLinear(z_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
        )
    
    def forward(self, x):
        return self.mapping(x)

class AdaIN(nn.Module):
    """Adaptive Instance Normalization module"""
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

class InjectNoise(nn.Module):
    """Module to inject noise into feature maps"""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        return x + (self.weight * noise)

class WSConv2d(nn.Module):
    """Weight-scaled 2D convolution for improved training stability"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # Initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class GenBlock(nn.Module):
    """Generator block with AdaIN and noise injection"""
    def __init__(self, in_channel, out_channel, w_dim):
        super(GenBlock, self).__init__()
        self.conv1 = WSConv2d(in_channel, out_channel)
        self.conv2 = WSConv2d(out_channel, out_channel)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.inject_noise1 = InjectNoise(out_channel)
        self.inject_noise2 = InjectNoise(out_channel)
        self.adain1 = AdaIN(out_channel, w_dim)
        self.adain2 = AdaIN(out_channel, w_dim)
        
    def forward(self, x, w):
        x = self.adain1(self.leaky(self.inject_noise1(self.conv1(x))), w)
        x = self.adain2(self.leaky(self.inject_noise2(self.conv2(x))), w)
        return x

class Generator(nn.Module):
    """StyleGAN Generator with progressive growing"""
    def __init__(self, z_dim, w_dim, in_channels, img_channels=3, factors=None):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        
        if factors is None:
            self.factors = [1, 1, 1, 1, 1/2, 1/4]
        else:
            self.factors = factors
            
        self.starting_cte = nn.Parameter(torch.ones(1, in_channels, 4, 4))
        self.map = MappingNetwork(z_dim, w_dim)
        
        self.initial_adain1 = AdaIN(in_channels, w_dim)
        self.initial_adain2 = AdaIN(in_channels, w_dim)
        self.initial_noise1 = InjectNoise(in_channels)
        self.initial_noise2 = InjectNoise(in_channels)
        self.initial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)

        self.initial_rgb = WSConv2d(
            in_channels, img_channels, kernel_size=1, stride=1, padding=0
        )
        
        self.prog_blocks, self.rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_rgb])
        )

        for i in range(len(self.factors)-1):
            conv_in_c = int(in_channels * self.factors[i])
            conv_out_c = int(in_channels * self.factors[i+1])
            self.prog_blocks.append(GenBlock(conv_in_c, conv_out_c, w_dim))
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))
    
    def fade_in(self, alpha, upscaled, generated):
        """Blend between two images based on alpha"""
        return torch.tanh(alpha * generated + (1-alpha) * upscaled)

    def forward(self, noise, alpha, steps):
        """
        Forward pass through the generator
        
        Args:
            noise: Input noise vector
            alpha: Fade-in factor for progressive growing
            steps: Current step/resolution level
            
        Returns:
            Image at the requested resolution
        """
        w = self.map(noise)
        x = self.initial_adain1(self.initial_noise1(self.starting_cte), w)
        x = self.initial_conv(x)
        out = self.initial_adain2(self.leaky(self.initial_noise2(x)), w)

        if steps == 0:
            return self.initial_rgb(out)
        
        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode='bilinear')
            out = self.prog_blocks[step](upscaled, w)

        final_upscaled = self.rgb_layers[steps-1](upscaled)
        final_out = self.rgb_layers[steps](out)

        return self.fade_in(alpha, final_upscaled, final_out)

class ConvBlock(nn.Module):
    """Discriminator convolutional block"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        return x

class Discriminator(nn.Module):
    """StyleGAN Discriminator with progressive growing"""
    def __init__(self, in_channels, img_channels=3, factors=None):
        super(Discriminator, self).__init__()
        
        if factors is None:
            self.factors = [1, 1, 1, 1, 1/2, 1/4]
        else:
            self.factors = factors
            
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        # Work backwards from factors for discriminator
        for i in range(len(self.factors) - 1, 0, -1):
            conv_in = int(in_channels * self.factors[i])
            conv_out = int(in_channels * self.factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out))
            self.rgb_layers.append(
                WSConv2d(img_channels, conv_in, kernel_size=1, stride=1, padding=0)
            )

        # RGB layer for 4x4 resolution
        self.initial_rgb = WSConv2d(
            img_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.rgb_layers.append(self.initial_rgb)
        
        # Downsampling with avg pooling
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Final block for 4x4 input
        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1),
        )

    def fade_in(self, alpha, downscaled, out):
        """Blend between two feature maps based on alpha"""
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        """Add minibatch standard deviation as a feature"""
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        """
        Forward pass through the discriminator
        
        Args:
            x: Input image
            alpha: Fade-in factor for progressive growing
            steps: Current step/resolution level
            
        Returns:
            Discrimination score
        """
        cur_step = len(self.prog_blocks) - steps

        # Convert from RGB
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:  # 4x4 resolution
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        # Fade-in for progressive growing
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)