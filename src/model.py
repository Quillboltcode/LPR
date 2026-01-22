import torch
import torch.nn as nn
import torchvision.models as models
import os

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_FRAMES = 5  # Each track has 5 frames
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3

# Character set - assuming alphanumeric license plates
import string
CHARACTERS = string.ascii_uppercase + string.digits
NUM_CLASSES = len(CHARACTERS) + 1  # +1 for CTC blank token


class TemporalCRNN(nn.Module):
    """
    Temporal CRNN (Convolutional Recurrent Neural Network) for license plate recognition.

    Architecture:
    1. Backbone: ResNet-18/34 (pre-trained) - for feature extraction from individual frames
    2. Temporal Fusion: Frame averaging for temporal feature aggregation (temporal super-resolution)
    3. Sequence Modeling: Bi-LSTM - for contextual character recognition
    4. Prediction: Fully connected layer with CTC loss

    Based on project plan requirements for ICPR 2026 submission.
    """

    def __init__(self, num_classes=NUM_CLASSES, backbone='resnet18', hidden_size=256):
        super(TemporalCRNN, self).__init__()

        # Backbone selection (ResNet-18 or ResNet-34 as per plan)
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights='DEFAULT')
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights='DEFAULT')
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet18' or 'resnet34'")

        # Remove the last pooling and FC layers to get feature maps
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Calculate backbone output dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
            dummy_output = self.backbone(dummy_input)
            self.backbone_output_size = dummy_output.shape[1]  # 512 for ResNet18/34
            self.feature_height = dummy_output.shape[2]  # 7 for 224x224 input
            self.feature_width = dummy_output.shape[3]   # 7 for 224x224 input

        # Temporal Fusion: Enhanced temporal aggregation
        # Use 1D convolution along temporal dimension for better feature fusion
        self.temporal_conv = nn.Conv1d(
            in_channels=self.backbone_output_size * self.feature_height * self.feature_width,
            out_channels=self.backbone_output_size * self.feature_height,
            kernel_size=NUM_FRAMES,
            stride=1,
            padding=0
        )

        # Sequence Modeling: Bi-LSTM
        # Use feature_width (7) as sequence length, backbone_output_size as feature dimension
        # This reduces input size from 3584 to 512, making it more efficient
        lstm_input_size = self.backbone_output_size
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2  # Add dropout for regularization
        )

        # Prediction: Fully connected layer to character classes
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        """
        Forward pass of the Temporal CRNN.

        Args:
            x: Input tensor of shape (batch_size, num_frames, channels, height, width)

        Returns:
            Output tensor of shape (batch, num_classes, sequence_length)
        """
        batch_size, num_frames, c, h, w = x.shape

        # Step 1: Process each frame independently through shared backbone
        x = x.view(batch_size * num_frames, c, h, w)
        features = self.backbone(x)  # Shape: (batch*frames, backbone_output_size, feature_height, feature_width)

        # Step 2: Reshape to (batch, frames, backbone_output_size, feature_height, feature_width)
        features = features.view(batch_size, num_frames, self.backbone_output_size, self.feature_height, self.feature_width)

        # Step 3: Temporal fusion - Enhanced temporal aggregation
        # Flatten spatial dimensions and apply temporal convolution
        b, f, c_feat, h_feat, w_feat = features.shape
        features_flat = features.view(b, f, -1)  # (batch, frames, c*h*w)
        features_flat = features_flat.permute(0, 2, 1)  # (batch, c*h*w, frames)

        # Apply temporal convolution for better temporal fusion
        temporal_fused = self.temporal_conv(features_flat)  # (batch, c*h, 1)
        temporal_fused = temporal_fused.squeeze(-1)  # (batch, c*h, 1) -> (batch, c*h)

        # Reshape for sequence modeling: (batch, feature_height, backbone_output_size)
        sequence_features = temporal_fused.view(b, self.feature_height, self.backbone_output_size)

        # Step 4: Sequence modeling with Bi-LSTM
        # sequence_features: (batch, seq_len=feature_height, input_size=backbone_output_size)
        lstm_out, _ = self.lstm(sequence_features)

        # Step 5: Prediction
        output = self.fc(lstm_out)  # (batch, seq_len, num_classes)

        # Step 6: Permute for CTC loss (batch, num_classes, sequence_length)
        output = output.permute(0, 2, 1)

        return output


def decode_prediction(output, characters=CHARACTERS):
    """
    Decode CTC output to text.
    
    Args:
        output: Model output tensor (shape: (num_classes, sequence_length))
        characters: Character set for decoding
        
    Returns:
        Decoded text string
    """
    # Get the most probable character at each step
    _, pred_indices = torch.max(output, dim=0)
    
    # Remove duplicates and blank tokens (0)
    decoded = []
    previous = 0
    for idx in pred_indices:
        idx = idx.item()
        if idx != previous and idx != 0:
            decoded.append(characters[idx - 1])
        previous = idx
    
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
