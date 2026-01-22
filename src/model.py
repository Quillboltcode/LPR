import os
import numpy as np
import torch
import torchaudio.models.decoder as decoder
import torch.nn as nn
import torchvision.models as models
import string
from src.FSRCNN import CBAM

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_FRAMES = 5

# Image sizes optimized for ResNet-18 (stride=32) and original aspect ratio
# Original LR: ~50x18 (aspect ~2.78:1), HR: ~129x45 (aspect ~2.87:1), scale ~2.5-2.6x
# Using 64x192 for better feature dimensions with ResNet:
# - Feature size: 512 x (64/32) x (192/32) = 512 x 2 x 6
# - LSTM input: 512 * 2 = 1024
# - Sequence length: 6 (for CTC decoding 7 characters)
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 192
HR_HEIGHT = 128
HR_WIDTH = 384
NUM_CHANNELS = 3

CHARACTERS = string.ascii_uppercase + string.digits
NUM_CLASSES = len(CHARACTERS) + 1  # +1 for CTC blank token

TARGET_LENGTH = 7

tokens = [''] + list(CHARACTERS)

beam_search_decoder = decoder.ctc_decoder(
    lexicon=None,
    tokens=tokens,
    blank_token='',
    sil_token='',
    lm=None,
    beam_size=50,
    nbest=50
)

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

        # Calculate dimensions based on input size (64x192)
        # ResNet stride is 32.
        # Feature Height = 64 / 32 = 2
        # Feature Width = 192 / 32 = 6
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
            dummy_output = self.backbone(dummy_input)
            self.backbone_output_size = dummy_output.shape[1]  # 512
            self.feature_height = dummy_output.shape[2]        # 2
            self.feature_width = dummy_output.shape[3]         # 6

        # --- CBAM Attention Module ---
        # Lightweight attention mechanism to focus on important features
        self.cbam = CBAM(self.backbone_output_size)

        # 3. Sequence Modeling Preparation
        # We need to collapse the spatial dimensions (H, C) into a single vector per step (Width).
        # The LSTM will read the image from Left to Right (Sequence Length = Feature Width).

        # Input size to LSTM = Channels * Height
        # For ResNet18 on 64x192 input: 512 * 2 = 1024
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

        # --- Step 1.5: Apply CBAM Attention ---
        # Focus on important spatial regions (the plate area)
        features = self.cbam(features)
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
    Decode CTC output (greedy search) with confidence metric.
    Args:
        output: Model output tensor (Time, Batch, Classes) OR (Time, Classes) if batch=1
    Returns:
        decoded_text: Decoded text string
        confidence: Average confidence score of the decoded characters
    """
    # Handle if single image passed (squeeze batch dim)
    if output.dim() == 3:
        output = output.squeeze(1) # (Time, Classes)
    
    # Softmax: Convert logits to probabilities
    probs = torch.softmax(output, dim=1)
    
    # Greedy Decode: Find character with highest probability at each time step
    max_probs, max_indices = torch.max(probs, dim=1)
    
    decoded = []
    survivor_probs = []
    previous_idx = -1 # Initialize to -1 to ensure first character is kept if not blank
    
    for idx, prob in zip(max_indices, max_probs):
        idx_val = idx.item()
        prob_val = prob.item()
        
        # CTC Rules:
        # 1. Ignore Blank (0)
        # 2. Ignore Repeated Characters
        if idx_val != 0 and idx_val != previous_idx:
            # Map index back to character (accounting for blank at 0)
            # Assuming standard encoding: 0=Blank, 1=A, 2=B...
            if idx_val - 1 < len(characters):
                decoded.append(characters[idx_val - 1])
                survivor_probs.append(prob_val)
        
        previous_idx = idx_val
    
    # Average: Calculate mean confidence of surviving characters
    confidence = np.mean(survivor_probs) if survivor_probs else 0.0
    
    return ''.join(decoded), confidence


def decode_with_beam_search_torchaudio(output_logits, target_length=TARGET_LENGTH):
    """
    Decodes using Torchaudio Beam Search and calculates confidence 
    based on score margin between best and second-best hypothesis.
    """
    # 1. Prepare Input
    if output_logits.dim() == 3:
        emissions = output_logits.permute(1, 0, 2)
    else:
        emissions = output_logits.unsqueeze(0)

    emissions = emissions.cpu()

    # 2. Run Beam Search
    results = beam_search_decoder(emissions)
    nbest_hypotheses = results[0]
    
    # 3. Filter hypotheses for the exact target length
    candidates = []
    for hypothesis in nbest_hypotheses:
        tokens_list = hypothesis.tokens
        score = hypothesis.score
        
        # Convert tokens to text
        text = ''.join([tokens[token_id] for token_id in tokens_list if token_id > 0])
        
        if len(text) == target_length:
            candidates.append((text, score))
    
    # 4. Handle Cases
    
    # Case A: No candidate found with correct length
    if not candidates:
        if len(nbest_hypotheses) > 0:
            # Fallback: Use the Top-1 result regardless of length
            h = nbest_hypotheses[0]
            text = ''.join([tokens[tid] for tid in h.tokens if tid > 0])
            return text, 0.0
        return "", 0.0

    # Sort candidates by score (descending, since score is log-prob)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    best_text, best_score = candidates[0]
    
    # Case B: Only one candidate found with correct length
    if len(candidates) == 1:
        # We have no competitor to compare against.
        # Fall back to probability-based method for this specific case
        confidence = np.exp(best_score / target_length)
        confidence = max(0.0, min(1.0, confidence))
        return best_text, confidence

    # Case C: Multiple candidates found (The Robust Case)
    # Calculate Margin: How much better is the best option vs. runner-up?
    runner_up_score = candidates[1][1]
    
    # The difference in log-space
    score_margin = best_score - runner_up_score 
    
    # Convert margin to confidence
    # If margin is large (e.g., 5.0), exp(5) is huge -> Clamp to 1.0
    # If margin is small (e.g., 0.1), exp(0.1) is ~1.1 -> Low confidence
    
    # Using a sigmoid-like approach for smoother 0-1 mapping or simple exp clamping:
    # Here we use 1 - e^(-margin) to scale it nicely
    confidence = 1.0 - np.exp(-score_margin)
    
    return best_text, confidence


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
    decoded_text, confidence = decode_prediction(test_output.squeeze(0))
    print(f"Decoded text: '{decoded_text}'")
    print(f"Confidence: {confidence:.4f}")

    # Test model with different hidden size
    model_large = TemporalCRNN(backbone='resnet18', hidden_size=512)
    print("Temporal CRNN (large hidden size) created successfully")

    print("All model configurations tested successfully!")
