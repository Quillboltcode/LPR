import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
import os
import argparse
from src.dataset import LicensePlateDataset, get_default_transform, get_lr_augmentation_transform
from src.model import TemporalCRNN, save_model_weights, load_model_weights, decode_prediction
from src.metric import calculate_recognition_rate, calculate_character_recognition_rate

# Character set
import string
CHARACTERS = string.ascii_uppercase + string.digits


def train_phase(model, train_dataloader, val_dataloader, optimizer, criterion, device, num_epochs, phase_name, patience=5):
    """
    Train the model for one phase.
    """
    best_val_recognition_rate = 0.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (frames, text_encoded, text_lengths, track_ids, _) in enumerate(train_dataloader):
            frames = frames.to(device)
            text_encoded = text_encoded.to(device)
            text_lengths = text_lengths.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(frames)

            # Prepare targets for CTC loss
            # Ensure log_softmax is applied on the correct dimension
            # Assuming model outputs (Time, Batch, Classes) based on previous correction
            log_probs = outputs.log_softmax(2)

            batch_size = outputs.size(1)
            # The length of the sequence is the Time dimension (dim 0)
            input_lengths = torch.full((batch_size,), outputs.size(0), dtype=torch.long)
            target_lengths = text_lengths.cpu()
            targets = text_encoded.view(-1).cpu()

            # Compute CTC loss
            # zero_infinity=True prevents NaNs if a path has zero probability
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            # Check for NaN loss before backward (sanity check)
            if torch.isnan(loss):
                print(f"NaN loss detected at Batch {batch_idx}. Skipping update.")
                continue

            # Backward pass
            loss.backward()

            # --- FIX 1: Gradient Clipping (CRITICAL) ---
            # This prevents exploding gradients common in LSTMs/CTC
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"{phase_name} - Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        # ... (Rest of validation logic remains the same) ...

        # Evaluate on validation set
        val_metrics = evaluate_model(model, val_dataloader, device)
        print(f"{phase_name} - Epoch {epoch+1}/{num_epochs} Validation:")
        print(f"  Recognition Rate: {val_metrics['recognition_rate']:.4f}")
        print(f"  Character Recognition Rate: {val_metrics['character_rate']:.4f}")

        # Early stopping check
        if val_metrics['recognition_rate'] > best_val_recognition_rate:
            best_val_recognition_rate = val_metrics['recognition_rate']
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation recognition rate.")
                break
        print(f"{phase_name} - Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")


def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on validation/test data.

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on

    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    predictions = {}
    ground_truths = {}

    with torch.no_grad():
        for frames, text_encoded, text_lengths, track_ids, _ in dataloader:
            frames = frames.to(device)
            outputs = model(frames)
            
            # Permute to (Batch, Time, Classes) for decoding
            outputs = outputs.permute(1, 0, 2)

            # Decode predictions
            for i, track_id in enumerate(track_ids):
                pred_output = outputs[i]
                pred_text, _ = decode_prediction(pred_output)
                predictions[track_id] = pred_text

                # Decode ground truth
                gt_indices = text_encoded[i][:text_lengths[i]].tolist()
                gt_text = ''.join([CHARACTERS[idx-1] for idx in gt_indices if idx > 0])
                ground_truths[track_id] = gt_text

    # Calculate metrics
    recognition_rate, correct_tracks, total_tracks = calculate_recognition_rate(predictions, ground_truths)
    char_rate, correct_chars, total_chars = calculate_character_recognition_rate(predictions, ground_truths)

    return {
        'recognition_rate': recognition_rate,
        'character_rate': char_rate,
        'correct_tracks': correct_tracks,
        'total_tracks': total_tracks,
        'correct_chars': correct_chars,
        'total_chars': total_chars
    }


def main():
    parser = argparse.ArgumentParser(description="Train Temporal CRNN for License Plate Recognition")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the data directory")
    parser.add_argument('--hr_epochs', type=int, default=10, help="Number of epochs for HR pre-training")
    parser.add_argument('--lr_epochs', type=int, default=20, help="Number of epochs for LR fine-tuning")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr_hr', type=float, default=1e-4, help="Learning rate for HR pre-training")
    parser.add_argument('--lr_lr', type=float, default=1e-4, help="Learning rate for LR fine-tuning")
    parser.add_argument('--weights_dir', type=str, default='weights', help="Directory to save/load weights")
    parser.add_argument('--skip_phase1', action='store_true', help="Skip Phase 1 and continue from Phase 2 using existing HR weights")
    parser.add_argument('--phase1_weights', type=str, default='weights/hr_pretrained_weights.pth', help="Path to Phase 1 weights for continuing to Phase 2")
    args = parser.parse_args()

    # Create weights directory
    os.makedirs(args.weights_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Character set
    NUM_CLASSES = len(CHARACTERS) + 1

    hr_weights_path = os.path.join(args.weights_dir, 'hr_pretrained_weights.pth')

    if args.skip_phase1:
        print(f"=== Skipping Phase 1, using existing HR weights from {args.phase1_weights} ===")
        phase1_weights_path = args.phase1_weights
        
        if not os.path.exists(phase1_weights_path):
            raise FileNotFoundError(f"Phase 1 weights not found at {phase1_weights_path}. Run training without --skip_phase1 first.")
        
        # Initialize model and load Phase 1 weights
        model = TemporalCRNN(num_classes=NUM_CLASSES).to(device)
        model = load_model_weights(model, phase1_weights_path)
        print(f"Loaded Phase 1 weights from {phase1_weights_path}")
    else:
        # Phase 1: HR Pre-training
        print("=== Phase 1: HR Pre-training ===")

        # Load HR datasets
        hr_train_dataset = LicensePlateDataset(
            tracks_dir=os.path.join(args.data_dir, 'train'),
            transform=get_default_transform(),
            is_lr=False,
            split='train'
        )
        hr_train_dataloader = DataLoader(hr_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        hr_val_dataset = LicensePlateDataset(
            tracks_dir=os.path.join(args.data_dir, 'train'),
            transform=get_default_transform(),
            is_lr=False,
            split='val'
        )
        hr_val_dataloader = DataLoader(hr_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Initialize model
        model = TemporalCRNN(num_classes=NUM_CLASSES).to(device)

        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=args.lr_hr)
        criterion = CTCLoss(blank=0, reduction='mean', zero_infinity=True)

        # Train on HR data
        train_phase(model, hr_train_dataloader, hr_val_dataloader, optimizer, criterion, device, args.hr_epochs, "HR Pre-training", patience=5)

        # Save HR weights
        save_model_weights(model, hr_weights_path)

        # Evaluate on HR validation data
        hr_metrics = evaluate_model(model, hr_val_dataloader, device)
        print(f"HR Pre-training Results:")
        print(f"  Recognition Rate: {hr_metrics['recognition_rate']:.4f}")
        print(f"  Character Recognition Rate: {hr_metrics['character_rate']:.4f}")

    # Phase 2: LR Fine-tuning
    print("\n=== Phase 2: LR Fine-tuning ===")

    # Load LR datasets with augmentation
    lr_train_dataset = LicensePlateDataset(
        tracks_dir=os.path.join(args.data_dir, 'train'),
        transform=get_lr_augmentation_transform(),
        is_lr=True,
        split='train'
    )
    lr_train_dataloader = DataLoader(lr_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    lr_val_dataset = LicensePlateDataset(
        tracks_dir=os.path.join(args.data_dir, 'train'),
        transform=get_lr_augmentation_transform(),
        is_lr=True,
        split='val'
    )
    lr_val_dataloader = DataLoader(lr_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load HR weights
    if not args.skip_phase1:
        model = load_model_weights(model, hr_weights_path)

    # Update optimizer with lower learning rate
    optimizer = optim.Adam(model.parameters(), lr=args.lr_lr)
    criterion = CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    # Fine-tune on LR data
    train_phase(model, lr_train_dataloader, lr_val_dataloader, optimizer, criterion, device, args.lr_epochs, "LR Fine-tuning", patience=5)

    # Save final weights
    final_weights_path = os.path.join(args.weights_dir, 'final_lr_finetuned_weights.pth')
    save_model_weights(model, final_weights_path)

    # Evaluate on LR validation data
    lr_metrics = evaluate_model(model, lr_val_dataloader, device)
    print(f"LR Fine-tuning Results:")
    print(f"  Recognition Rate: {lr_metrics['recognition_rate']:.4f}")
    print(f"  Character Recognition Rate: {lr_metrics['character_rate']:.4f}")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()