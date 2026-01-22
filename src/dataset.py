import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import string
import random
import numpy as np
from PIL import ImageFilter

# Character set - assuming alphanumeric license plates
CHARACTERS = string.ascii_uppercase + string.digits
NUM_CLASSES = len(CHARACTERS) + 1  # +1 for CTC blank token

# Configuration
NUM_FRAMES = 5  # Each track has 5 frames
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 192 # Optimized for original aspect ratio and ResNet-18 stride
NUM_CHANNELS = 3


class LicensePlateDataset(Dataset):
    """
    Dataset for loading license plate tracks with multiple frames and their annotations.
    
    A track consists of multiple frames of the same license plate, along with
    annotations including plate text and corner coordinates for each frame.
    
    Args:
        tracks_dir (str): Path to the directory containing track subdirectories.
        transform (callable, optional): Optional transform to be applied on each image frame.
        num_frames (int): Number of frames to load per track (default: 5).
        is_lr (bool): Whether the dataset is for low-resolution training (default: False).
    """

    def __init__(self, tracks_dir, transform=None, num_frames=NUM_FRAMES, is_lr=False, split='train', val_split=0.2):
        self.tracks_dir = tracks_dir
        self.transform = transform
        self.num_frames = num_frames
        self.is_lr = is_lr
        self.split = split
        self.val_split = val_split
        self.tracks = self._load_tracks()
        self._split_tracks()
    
    def _load_tracks(self):
        """
        Load track information from the dataset directory.

        Returns:
            list: List of track dictionaries containing track ID, image files, track path,
                  plate text, and corner coordinates for each frame.
        """
        tracks = []

        # Recursively find all track directories (those containing annotations.json)
        for root, dirs, files in os.walk(self.tracks_dir):
            if 'annotations.json' in files:
                track_path = root
                annotations_file = os.path.join(track_path, 'annotations.json')

                if os.path.exists(annotations_file):
                    with open(annotations_file, 'r', encoding='utf-8') as f:
                        annotations = json.load(f)

                    # Find all image files in the track directory
                    image_files = []
                    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                        if self.is_lr:
                            image_files.extend([f for f in os.listdir(track_path) if f.endswith(ext) and f.startswith('lr-')])
                        else:
                            image_files.extend([f for f in os.listdir(track_path) if f.endswith(ext) and f.startswith('hr-')])

                    # Sort image files (assuming names like lr-001.jpg, lr-002.jpg, etc.)
                    image_files.sort()

                    if len(image_files) >= self.num_frames:
                        tracks.append({
                            'id': os.path.relpath(track_path, self.tracks_dir),
                            'image_files': image_files,
                            'track_path': track_path,
                            'plate_text': annotations['plate_text'],
                            'plate_layout': annotations['plate_layout'],
                            'corners': annotations['corners']
                        })
        
        return tracks

    def _split_tracks(self):
        """
        Split tracks into train and validation sets.
        """
        # Sort by ID to ensure deterministic order before shuffling
        self.tracks.sort(key=lambda x: x['id'])
        # Use local Random instance for reproducibility without affecting global state
        rng = random.Random(42)
        rng.shuffle(self.tracks)
        split_idx = int(len(self.tracks) * (1 - self.val_split))
        if self.split == 'train':
            self.tracks = self.tracks[:split_idx]
        elif self.split == 'val':
            self.tracks = self.tracks[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'val'")
    
    def _encode_text(self, text):
        """
        Encode plate text to list of character indices.
        
        Args:
            text (str): Plate text to encode.
            
        Returns:
            list: List of character indices (1-based, 0 reserved for CTC blank token).
        """
        return [CHARACTERS.index(c) + 1 for c in text if c in CHARACTERS]
    
    def __len__(self):
        """
        Get the number of tracks in the dataset.
        
        Returns:
            int: Number of tracks.
        """
        return len(self.tracks)
    
    def __getitem__(self, idx):
        """
        Get a track from the dataset.
        
        Args:
            idx (int): Index of the track to retrieve.
            
        Returns:
            tuple: (frames, text_encoded, text_length, track_id, corners)
                - frames: Tensor of shape (num_frames, channels, height, width)
                - text_encoded: Tensor of encoded plate text
                - text_length: Length of the encoded text
                - track_id: Unique identifier for the track
                - corners: JSON string of dictionary mapping image filenames to corner coordinates
        """
        track = self.tracks[idx]
        
        # Sample frames
        image_files = track['image_files']
        if self.split == 'train':
            # Randomly sample a sequence of frames
            max_start = len(image_files) - self.num_frames
            start_idx = random.randint(0, max_start)
        else:
            # Deterministic sampling (center crop) for validation
            start_idx = (len(image_files) - self.num_frames) // 2
            
        selected_files = image_files[start_idx : start_idx + self.num_frames]

        # Load all frames for the track
        frames = []
        for img_file in selected_files:
            img_path = os.path.join(track['track_path'], img_file)
            image = Image.open(img_path).convert('RGB')

            # Apply data augmentation for LR training
            if self.is_lr:
                # Apply Gaussian blur
                image = apply_gaussian_blur(image)

            if self.transform:
                image = self.transform(image)

            # Apply Gaussian noise after tensor conversion
            if self.is_lr:
                image = apply_gaussian_noise(image)

            frames.append(image)
        
        # Stack frames into a single tensor
        frames = torch.stack(frames)
        
        # Encode plate text
        text_encoded = self._encode_text(track['plate_text'])
        text_length = torch.tensor(len(text_encoded), dtype=torch.long)

        return frames, torch.tensor(text_encoded, dtype=torch.long), text_length, track['id'], json.dumps(track['corners'])


def get_default_transform():
    """
    Get default image transforms (resize and convert to tensor).
    
    Returns:
        transforms.Compose: Default transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_lr_augmentation_transform():
    """
    Get image transforms with data augmentation for low-resolution training.
    
    Returns:
        transforms.Compose: Augmented transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def apply_gaussian_noise(image_tensor):
    """
    Apply Gaussian noise to a tensor image.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor.
        
    Returns:
        torch.Tensor: Image tensor with Gaussian noise.
    """
    noise = torch.randn_like(image_tensor) * 0.1
    noisy_image = image_tensor + noise
    return torch.clamp(noisy_image, 0.0, 1.0)


def apply_gaussian_blur(image):
    """
    Apply Gaussian blur to a PIL image.
    
    Args:
        image (PIL.Image): Input PIL image.
        
    Returns:
        PIL.Image: Blurred image.
    """
    return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))


if __name__ == "__main__":
    # Test script
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Test LicensePlateDataset")
    parser.add_argument('--tracks_dir', type=str, required=True, help="Path to tracks directory")
    args = parser.parse_args()
    
    # Create dataset with default transforms
    dataset = LicensePlateDataset(
        tracks_dir=args.tracks_dir,
        transform=get_default_transform()
    )

    print(f"Dataset size: {len(dataset)} tracks")
    print()

    # Test loading all tracks
    for i in range(len(dataset)):
        try:
            frames, text_encoded, text_length, track_id, corner_coords_json = dataset[i]
            corner_coords = json.loads(corner_coords_json)
            print(f"Track {i+1}: {track_id}")
            print(f"  Text Encoded: {text_encoded}")
            print(f"  Text Length: {text_length}")
            print(f"  Frames Shape: {frames.shape}")
            print(f"  Corner Coordinates: {list(corner_coords.keys())}")
            for img_file, coords in corner_coords.items():
                if coords:
                    if isinstance(coords, list):
                        print(f"    {img_file}: {[int(x) for x in coords]}")
                    else:
                        print(f"    {img_file}: {coords}")
                else:
                    print(f"    {img_file}: No corner coordinates")
            print()
        except Exception as e:
            print(f"Error loading track {i}: {e}")
            print()
        break
    # Visualize the first frame of the first track (without normalization for display)
    print("Visualizing first frame of first track...")
    dataset_no_norm = LicensePlateDataset(
        tracks_dir=args.tracks_dir,
        transform=transforms.Compose([
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor()
        ])
    )

    frames, text_encoded, text_length, track_id, corner_coords_json = dataset_no_norm[0]
    corner_coords = json.loads(corner_coords_json)
    
    # Display first frame
    plt.imshow(frames[0].permute(1, 2, 0))
    
    # Get corner coordinates for first frame
    first_frame_file = dataset_no_norm.tracks[0]['image_files'][0]
    if first_frame_file in corner_coords and corner_coords[first_frame_file]:
        coords = corner_coords[first_frame_file]
        x_coords = []
        y_coords = []
        
        if isinstance(coords, list):
            corners = [int(x) for x in coords]
            x_coords = corners[0::2]
            y_coords = corners[1::2]
        elif isinstance(coords, dict):
            # Handle dictionary format (e.g. {'top-left': [x,y], ...})
            for key in ['top-left', 'top-right', 'bottom-right', 'bottom-left']:
                if key in coords:
                    x_coords.append(coords[key][0])
                    y_coords.append(coords[key][1])
        
        if x_coords and y_coords:
            # Close the polygon
            x_coords.append(x_coords[0])
            y_coords.append(y_coords[0])
            plt.plot(x_coords, y_coords, 'r-', linewidth=2)
    
    plt.title(f"Track: {track_id}\nText Encoded: {text_encoded}")
    plt.axis('off')
    
    plt.savefig('test_license_plate_dataset.png', bbox_inches='tight', dpi=300)
    print(f"Visualization saved to test_license_plate_dataset.png")
