import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import glob
from PIL import Image
import torchvision.transforms as transforms
from src.FSRCNN import FSRCNN


class SRDataset(Dataset):
    """
    Dataset for Super-Resolution training using HR/LR image pairs.
    """
    def __init__(self, data_dir, split='train', transform_hr=None, transform_lr=None):
        self.data_dir = data_dir
        self.split = split
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr
        
        self.pairs = self._load_pairs()
        
    def _load_pairs(self):
        """Load HR/LR image pairs from track folders."""
        pairs = []
        
        train_dir = os.path.join(self.data_dir, 'train')
        
        for track_path in glob.glob(os.path.join(train_dir, '*/*/track_*')):
            # Get base name
            track_name = os.path.basename(track_path)
            
            # Find matching HR and LR images
            lr_files = sorted(glob.glob(os.path.join(track_path, 'lr-*.png'))) + \
                      sorted(glob.glob(os.path.join(track_path, 'lr-*.jpg'))) + \
                      sorted(glob.glob(os.path.join(track_path, 'lr-*.jpeg'))) + \
                      sorted(glob.glob(os.path.join(track_path, 'lr-*.bmp')))
            
            hr_files = sorted(glob.glob(os.path.join(track_path, 'hr-*.png'))) + \
                      sorted(glob.glob(os.path.join(track_path, 'hr-*.jpg'))) + \
                      sorted(glob.glob(os.path.join(track_path, 'hr-*.jpeg'))) + \
                      sorted(glob.glob(os.path.join(track_path, 'hr-*.bmp')))
            
            # Pair up images by index
            for i in range(min(len(lr_files), len(hr_files))):
                pairs.append((lr_files[i], hr_files[i]))
        
        # Split into train/val
        if len(pairs) > 0:
            split_idx = int(len(pairs) * 0.8)
            if self.split == 'train':
                pairs = pairs[:split_idx]
            elif self.split == 'val':
                pairs = pairs[split_idx:]
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        lr_path, hr_path = self.pairs[idx]
        
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        if self.transform_lr:
            lr_img = self.transform_lr(lr_img)
        if self.transform_hr:
            hr_img = self.transform_hr(hr_img)
        
        return lr_img, hr_img


def get_sr_transforms():
    """Get transforms for SR training."""
    # Image sizes optimized for ResNet-18 (stride=32) and original aspect ratio
    # Original LR: ~50x18 (aspect ~2.78:1), HR: ~129x45 (aspect ~2.87:1), scale ~2.5-2.6x
    # Using 64x192 for LR to maintain aspect ratio and ensure proper feature dimensions
    transform_lr = transforms.Compose([
        transforms.Resize((64, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # HR is 2x LR for FSRCNN training (128x384)
    transform_hr = transforms.Compose([
        transforms.Resize((128, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform_lr, transform_hr


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        sr_imgs = model(lr_imgs)
        
        # Calculate L1 loss
        loss = criterion(sr_imgs, hr_imgs)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for lr_imgs, hr_imgs in val_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train FSRCNN for Super-Resolution")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the data directory")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weights_dir', type=str, default='weights', help="Directory to save weights")
    args = parser.parse_args()
    
    # Create weights directory
    os.makedirs(args.weights_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get transforms
    transform_lr, transform_hr = get_sr_transforms()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = SRDataset(
        data_dir=args.data_dir,
        split='train',
        transform_lr=transform_lr,
        transform_hr=transform_hr
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    val_dataset = SRDataset(
        data_dir=args.data_dir,
        split='val',
        transform_lr=transform_lr,
        transform_hr=transform_hr
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Initialize model
    print("\nInitializing FSRCNN model...")
    model = FSRCNN(in_channels=3, out_channels=3, scale_factor=2, num_features=32, num_map_layers=12).to(device)
    
    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            weights_path = os.path.join(args.weights_dir, 'fsrcnn_best.pth')
            torch.save(model.state_dict(), weights_path)
            print(f"New best model saved to {weights_path} (Val Loss: {val_loss:.4f})")
    
    # Save final model
    final_weights_path = os.path.join(args.weights_dir, 'fsrcnn_final.pth')
    torch.save(model.state_dict(), final_weights_path)
    print(f"\nFinal model saved to {final_weights_path}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
