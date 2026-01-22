import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
import os
from src.dataset import LicensePlateDataset, get_default_transform
from src.model import TemporalCRNN
import string

# Character set
CHARACTERS = string.ascii_uppercase + string.digits
NUM_CLASSES = len(CHARACTERS) + 1

def test_one_batch():
    """Test training with one batch."""
    print("Testing Temporal CRNN with one batch...")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load a small dataset (just for testing)
    dataset = LicensePlateDataset(
        tracks_dir='data/train/Scenario-A/Brazilian',
        transform=get_default_transform(),
        is_lr=False
    )

    # Create dataloader with batch size 1
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize model
    model = TemporalCRNN(num_classes=NUM_CLASSES).to(device)
    print("Model created successfully")

    # Get one batch
    try:
        frames, text_encoded, text_lengths, track_ids, _ = next(iter(dataloader))
        print(f"Loaded batch with shape: {frames.shape}")
        print(f"Track ID: {track_ids[0]}")
        print(f"Text encoded: {text_encoded}")
        print(f"Text length: {text_lengths}")

        frames = frames.to(device)
        text_encoded = text_encoded.to(device)
        text_lengths = text_lengths.to(device)

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(frames)
            print(f"Model output shape: {outputs.shape}")

        # Test training step
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = CTCLoss(blank=0, reduction='mean')

        optimizer.zero_grad()

        # Forward pass
        outputs = model(frames)

        # Prepare targets for CTC loss
        # CTC expects log_probs of shape (T, N, C) where T=seq_len, N=batch_size, C=num_classes
        log_probs = outputs.permute(2, 0, 1)  # (7, 1, 37)

        batch_size = outputs.size(0)
        input_lengths = torch.full((batch_size,), outputs.size(2), dtype=torch.long)  # [7] - on CPU
        target_lengths = text_lengths.cpu()  # [7] - on CPU

        # Flatten targets - CTC expects 1D tensor with concatenated targets
        targets = text_encoded.view(-1).cpu()  # [1, 24, 5, 27, 34, 32, 28] - on CPU

        print(f"log_probs shape: {log_probs.shape}")
        print(f"targets: {targets}")
        print(f"input_lengths: {input_lengths}")
        print(f"target_lengths: {target_lengths}")

        # Compute CTC loss
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        print(f"CTC Loss: {loss.item():.4f}")

        # Backward pass
        loss.backward()
        optimizer.step()

        print("Training step completed successfully!")
        print("Test passed!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_one_batch()