import os
import torch
import torch.nn as nn
import torchvision.models as models
import string

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_FRAMES = 5

# Use a rectangular aspect ratio (e.g., 3x1 ratio) suitable for license plates
IMAGE_HEIGHT = 64  
IMAGE_WIDTH = 256 # Increased width ensures longer sequence for CTC
NUM_CHANNELS = 3

CHARACTERS = string.ascii_uppercase + string.digits
NUM_CLASSES = len(CHARACTERS) + 1  # +1 for CTC blank token

class TemporalCRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, backbone='resnet18', hidden_size=256):
        super(TemporalCRNN, self).__init__()

        # 1. Backbone: ResNet-18
        if backbone == 'resnet18':
            resnet = models.resnet18(weights='DEFAULT')
        elif backbone == 'resnet34':
            resnet = models.resnet34(weights='DEFAULT')
        else:
            raise ValueError("Unsupported backbone")

        # Remove AvgPool and FC layer. 
        # Output of layer4 is (Batch, 512, H/32, W/32).
        # We remove the final pooling to keep the feature map grid.
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # 2. Temporal Fusion Strategy: Averaging
        # We will average the feature maps of the 5 frames manually in forward pass.
        # No learned parameters needed for this baseline approach (robust & simple).

        # Calculate dimensions based on new input size (64x256)
        # ResNet stride is 32. 
        # Feature Height = 64 / 32 = 2
        # Feature Width = 256 / 32 = 8
        # Note: 8 is short. If you have compute, reduce stride in layer4 or use a wider input.
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
            dummy_output = self.backbone(dummy_input)
            self.backbone_output_size = dummy_output.shape[1]  # 512
            self.feature_height = dummy_output.shape[2]        # 2
            self.feature_width = dummy_output.shape[3]         # 8

        # 3. Sequence Modeling Preparation
        # We need to collapse the spatial dimensions (H, C) into a single vector per step (Width).
        # The LSTM will read the image from Left to Right (Sequence Length = Feature Width).
        
        # Input size to LSTM = Channels * Height
        # For ResNet18 on 64x256 input: 512 * 2 = 1024
        self.lstm_input_size = self.backbone_output_size * self.feature_height

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )

        # 4. Prediction Layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, num_frames, channels, height, width)
        Returns:
            log_probs: Tensor of shape (Time, Batch, Num_Classes) for PyTorch CTC Loss
        """
        batch_size, num_frames, c, h, w = x.shape

        # --- Step 1: Extract Features (Shared Backbone) ---
        # Reshape to process all frames in one go
        x = x.view(batch_size * num_frames, c, h, w)
        features = self.backbone(x) 
        # Shape: (batch*frames, 512, feat_h, feat_w)

        # --- Step 2: Temporal Fusion (Averaging) ---
        # Reshape back to (batch, frames, channels, h, w)
        _, c, h, w = features.shape
        features = features.view(batch_size, num_frames, c, h, w)
        
        # Average across the temporal dimension (dim=1)
        # This acts as "temporal super-resolution", denoising the image
        fused_features = features.mean(dim=1) 
        # Shape: (batch, 512, feat_h, feat_w)

        # --- Step 3: Prepare Sequence for LSTM ---
        # We want to read Left-to-Right. So sequence length is `feature_width`.
        # We need to permute dimensions to (Batch, Width, Channels * Height)
        
        # Current: (Batch, Channel, Height, Width)
        # Target: (Batch, Width, Channel * Height)
        b, c_feat, h_feat, w_feat = fused_features.shape
        
        # Move Width to dim 1, collapse Channel and Height into dim 2
        sequence_input = fused_features.permute(0, 3, 1, 2) # (Batch, Width, Channel, Height)
        sequence_input = sequence_input.contiguous().view(b, w_feat, -1) # (Batch, Width, C*H)

        # --- Step 4: LSTM Processing ---
        lstm_out, _ = self.lstm(sequence_input)
        # Shape: (Batch, Width, Hidden*2)

        # --- Step 5: Prediction ---
        output = self.fc(lstm_out)
        # Shape: (Batch, Width, Num_Classes)

        # --- Step 6: Format for CTC ---
        # PyTorch CTC Loss expects (Time, Batch, Classes)
        # Width is our Time dimension
        output = output.permute(1, 0, 2) 
        
        return output

def decode_prediction(output, characters=CHARACTERS):
    """
    Decode CTC output (greedy search).
    Args:
        output: Model output tensor (Time, Batch, Classes) OR (Time, Classes) if batch=1
    """
    # Handle if single image passed (squeeze batch dim)
    if output.dim() == 3:
        output = output.squeeze(1) # (Time, Classes)
    
    # Get max probability index at each time step
    _, max_indices = torch.max(output, dim=1)
    
    decoded = []
    previous_idx = -1 # Initialize to -1 to ensure first character is kept if not blank
    
    for idx in max_indices:
        idx_val = idx.item()
        
        # CTC Rules:
        # 1. Ignore Blank (0)
        # 2. Ignore Repeated Characters
        if idx_val != 0 and idx_val != previous_idx:
            # Map index back to character (accounting for blank at 0)
            # Note: If Blank is 0, 'A' is 1. But in our list 'A' is index 0.
            # Check your encoding. Assuming Label 'A' -> 1, 'B' -> 2 ...
            # If your labels are encoded starting at 1 (0 is blank), subtract 1.
            # If your labels are encoded starting at 0 (0 is blank), then mapping is needed.
            # Assuming standard encoding: 0=Blank, 1=A, 2=B...
            if idx_val - 1 < len(characters):
                decoded.append(characters[idx_val - 1])
        
        previous_idx = idx_val
        
    return ''.join(decoded)


def save_model_weights(model, filepath):
    """
    Save model weights to a file.

    Args:
        model: PyTorch model
        filepath: Path to save the weights
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model weights saved to {filepath}")


def load_model_weights(model, filepath):
    """
    Load model weights from a file.

    Args:
        model: PyTorch model
        filepath: Path to load the weights from

    Returns:
        model: Model with loaded weights
    """
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath))
        print(f"Model weights loaded from {filepath}")
    else:
        print(f"Warning: Weights file {filepath} not found")
    return model


if __name__ == "__main__":
    # Test model initialization with different configurations
    print("Testing Temporal CRNN configurations...")

    # Test ResNet-18 (default)
    model18 = TemporalCRNN(backbone='resnet18')
    print("Temporal CRNN (ResNet-18) created successfully")

    # Test ResNet-34
    model34 = TemporalCRNN(backbone='resnet34')
    print("Temporal CRNN (ResNet-34) created successfully")

    # Test forward pass
    test_input = torch.randn(1, NUM_FRAMES, NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    test_output = model18(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")

    # Test decoding
    decoded_text = decode_prediction(test_output.squeeze(0))
    print(f"Decoded text: '{decoded_text}'")

    # Test model with different hidden size
    model_large = TemporalCRNN(backbone='resnet18', hidden_size=512)
    print("Temporal CRNN (large hidden size) created successfully")

    print("All model configurations tested successfully!")
