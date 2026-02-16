import os
import sys
sys.path.insert(0, 'src')
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import TemporalCRNN, decode_with_beam_search_torchaudio, CHARACTERS, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, DEVICE

def get_lr_test_transform():
    """
    Get image transforms for LR test images.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_track_images(track_path, transform):
    """
    Load lr- prefixed images from a track folder.
    
    Args:
        track_path: Path to the track folder
        transform: Image transforms to apply
        
    Returns:
        Tensor of shape (num_frames, channels, height, width)
    """
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(track_path, f'lr-{ext}')))
    image_files.sort()
    
    if len(image_files) == 0:
        raise ValueError(f"No lr- prefixed images found in {track_path}")
    
    images = []
    for img_file in image_files[:NUM_FRAMES]:
        image = Image.open(img_file).convert('RGB')
        image = transform(image)
        images.append(image)
    
    while len(images) < NUM_FRAMES:
        images.append(images[-1])
    
    return torch.stack(images)

def main():
    test_data_dir = 'data/test'
    weights_path = 'weights/failed_format/final_lr_finetuned_weights.pth'
    output_file = 'submission.csv'
    
    model = TemporalCRNN(backbone='resnet18').to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    
    transform = get_lr_test_transform()
    
    track_folders = sorted(glob.glob(os.path.join(test_data_dir, 'track_*')))
    
    print(f"Found {len(track_folders)} track folders in {test_data_dir}")
    if len(track_folders) != 1000:
        print(f"Warning: Expected 1000 tracks, but found {len(track_folders)}")
    
    results = []
    
    with torch.no_grad():
        for track_folder in track_folders:
            track_id = os.path.basename(track_folder)
            
            try:
                frames = load_track_images(track_folder, transform)
                frames = frames.unsqueeze(0).to(DEVICE)
                
                output = model(frames)
                decoded_text, confidence = decode_with_beam_search_torchaudio(output.squeeze(0))
                
                results.append((track_id, decoded_text, f"{confidence:.4f}"))
                print(f"{track_id}: {decoded_text} (confidence: {confidence:.4f})")
            except Exception as e:
                print(f"Error processing {track_id}: {e}")
                results.append((track_id, "", "0.0000"))
    
    with open(output_file, 'w') as f:
        for track_id, text, conf in results:
            f.write(f"{track_id},{text},{conf}\n")
    
    print(f"\nSubmission saved to {output_file}")
    print(f"Total tracks processed: {len(results)}/1000")
    if len(results) == 1000:
        print("All 1000 test samples processed successfully!")
    else:
        print(f"Warning: Only processed {len(results)} out of 1000 expected samples")

if __name__ == "__main__":
    main()
