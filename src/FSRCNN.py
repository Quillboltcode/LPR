import torch
import torch.nn as nn


class FSRCNN(nn.Module):
    """
    FSRCNN Architecture for Fast Super-Resolution.
    
    Structure:
    1. Feature Extraction: Conv(5x5)
    2. Shrinking: Conv(1x1) - Reduces channels for speed
    3. Mapping: Multiple Conv(3x3) - The core non-linear layers
    4. Expanding: Conv(1x1) - Restores channels
    5. Deconvolution: PixelShuffle - Upscales spatial size
    """
    def __init__(self, in_channels=3, out_channels=3, scale_factor=2, num_features=32, num_map_layers=12):
        super(FSRCNN, self).__init__()
        
        self.scale_factor = scale_factor
        
        # 1. Feature Extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=5, padding=2),
            nn.PReLU(num_features)
        )
        
        # 2. Shrinking (Reduces dimensionality)
        self.shrinking = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=1),
            nn.PReLU(num_features // 2)
        )
        
        # 3. Mapping (The "Deep" part)
        # Creates a stack of non-linear mapping layers
        map_layers = []
        for _ in range(num_map_layers):
            map_layers.append(nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1))
            map_layers.append(nn.PReLU(num_features // 2))
        self.mapping = nn.Sequential(*map_layers)
        
        # 4. Expanding (Restores dimensionality)
        self.expanding = nn.Sequential(
            nn.Conv2d(num_features // 2, num_features, kernel_size=1),
            nn.PReLU(num_features)
        )
        
        # 5. Deconvolution (Upsampling)
        # PixelShuffle is cleaner than Transposed Conv for text
        self.deconv = nn.Sequential(
            nn.Conv2d(num_features, out_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: Low Res Image (B, C, H, W)
        Returns:
            x: Super Res Image (B, C, H*scale, W*scale)
        """
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.deconv(x)
        return x


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling path
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        
        # Max pooling path
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        out = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        # Channel-wise pooling along channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        out = torch.cat([avg_out, max_out], dim=1)
        out = torch.sigmoid(self.conv(out))
        
        return x * out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Combines Channel Attention and Spatial Attention.
    """
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel)
        
    def forward(self, x):
        # Apply channel attention first
        x = self.channel_attention(x)
        
        # Apply spatial attention
        x = self.spatial_attention(x)
        
        return x
