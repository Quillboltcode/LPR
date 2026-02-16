import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import glob
import random
from PIL import Image
import torchvision.transforms as transforms
from src.diffplate import DiffPlateUNet, DiffPlate


class SRDataset(Dataset):
    def __init__(self, data_dir, split='train', transform_hr=None, transform_lr=None, val_split=0.2):
        self.data_dir = data_dir
        self.split = split
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr
        self.val_split = val_split
        
        self.pairs = self._load_pairs()
        self._split_pairs()
        
    def _load_pairs(self):
        pairs = []
        
        for track_path in glob.glob(os.path.join(self.data_dir, '**', 'track_*'), recursive=True):
            lr_files = sorted(glob.glob(os.path.join(track_path, 'lr-*.png')))
            hr_files = sorted(glob.glob(os.path.join(track_path, 'hr-*.png')))
            
            for i in range(min(len(lr_files), len(hr_files))):
                pairs.append((lr_files[i], hr_files[i]))
        
        return pairs
    
    def _split_pairs(self):
        self.pairs.sort(key=lambda x: x[0])
        random.Random(42).shuffle(self.pairs)
        
        split_idx = int(len(self.pairs) * (1 - self.val_split))
        if self.split == 'train':
            self.pairs = self.pairs[:split_idx]
        elif self.split == 'val':
            self.pairs = self.pairs[split_idx:]
        
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
    transform_lr = transforms.Compose([
        transforms.Resize((32, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.251, 0.251, 0.251], std=[0.324, 0.323, 0.319])
    ])
    
    transform_hr = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.251, 0.251, 0.251], std=[0.324, 0.323, 0.319])
    ])
    
    return transform_lr, transform_hr


def train_epoch(diff_plate, train_loader, optimizer, device):
    diff_plate.model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        
        optimizer.zero_grad()
        
        loss = diff_plate(hr_imgs, lr_imgs)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diff_plate.model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(diff_plate, val_loader, device):
    diff_plate.model.eval()
    total_loss = 0.0
    
    for lr_imgs, hr_imgs in val_loader:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        
        loss = diff_plate(hr_imgs, lr_imgs)
        total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train DiffPlate for Super-Resolution")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the data directory")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument('--noise_steps', type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument('--weights_dir', type=str, default='weights', help="Directory to save weights")
    parser.add_argument('--patience', type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()
    
    os.makedirs(args.weights_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform_lr, transform_hr = get_sr_transforms()
    
    print("Loading datasets...")
    train_dataset = SRDataset(
        data_dir=args.data_dir,
        split='train',
        transform_lr=transform_lr,
        transform_hr=transform_hr
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = SRDataset(
        data_dir=args.data_dir,
        split='val',
        transform_lr=transform_lr,
        transform_hr=transform_hr
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Training config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Diffusion config: noise_steps={args.noise_steps}")
    
    print("\nInitializing DiffPlate model...")
    unet = DiffPlateUNet(image_height=64, image_width=128)
    diff_plate = DiffPlate(
        unet, 
        image_height=64, 
        image_width=128, 
        device=str(device)
    )
    diff_plate.noise_steps = args.noise_steps
    
    optimizer = optim.AdamW(
        diff_plate.model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in diff_plate.model.parameters()):,}")
    print(f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"Noise steps: {args.noise_steps}")
    
    print("\nStarting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    last_epoch = 0
    last_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        train_loss = train_epoch(diff_plate, train_loader, optimizer, device)
        val_loss = validate(diff_plate, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            weights_path = os.path.join(args.weights_dir, 'diffplate_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': diff_plate.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, weights_path)
            print(f"New best model saved to {weights_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        last_epoch = epoch
        last_val_loss = val_loss
    
    final_weights_path = os.path.join(args.weights_dir, 'diffplate_final.pth')
    torch.save({
        'epoch': last_epoch,
        'model_state_dict': diff_plate.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': last_val_loss,
    }, final_weights_path)
    print(f"\nFinal model saved to {final_weights_path}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
