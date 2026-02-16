import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict
import argparse


def analyze_dataset(data_dir, sample_limit=None):
    lr_sizes = []
    hr_sizes = []
    lr_pixel_values = []
    hr_pixel_values = []
    
    track_count = 0
    for root, dirs, files in os.walk(data_dir):
        if 'annotations.json' in files:
            track_path = root
            
            lr_files = [f for f in files if f.startswith('lr-') and f.endswith('.png')]
            hr_files = [f for f in files if f.startswith('hr-') and f.endswith('.png')]
            
            for lr_file in sorted(lr_files):
                img_path = os.path.join(track_path, lr_file)
                img = Image.open(img_path)
                lr_sizes.append(img.size)
                img_array = np.array(img.convert('RGB')).astype(np.float32) / 255.0
                lr_pixel_values.append(img_array)
            
            for hr_file in sorted(hr_files):
                img_path = os.path.join(track_path, hr_file)
                img = Image.open(img_path)
                hr_sizes.append(img.size)
                img_array = np.array(img.convert('RGB')).astype(np.float32) / 255.0
                hr_pixel_values.append(img_array)
            
            track_count += 1
            if sample_limit and track_count >= sample_limit:
                break
    
    return analyze_sizes(lr_sizes, hr_sizes), analyze_pixel_stats(lr_pixel_values, hr_pixel_values), track_count


def analyze_sizes(lr_sizes, hr_sizes):
    lr_widths = [s[0] for s in lr_sizes]
    lr_heights = [s[1] for s in lr_sizes]
    hr_widths = [s[0] for s in hr_sizes]
    hr_heights = [s[1] for s in hr_sizes]
    
    return {
        'lr': {
            'width': {'min': min(lr_widths), 'max': max(lr_widths), 'mean': np.mean(lr_widths), 'std': np.std(lr_widths), 'median': np.median(lr_widths)},
            'height': {'min': min(lr_heights), 'max': max(lr_heights), 'mean': np.mean(lr_heights), 'std': np.std(lr_heights), 'median': np.median(lr_heights)},
            'count': len(lr_sizes),
            'unique_sizes': list(set(lr_sizes))[:20],
        },
        'hr': {
            'width': {'min': min(hr_widths), 'max': max(hr_widths), 'mean': np.mean(hr_widths), 'std': np.std(hr_widths), 'median': np.median(hr_widths)},
            'height': {'min': min(hr_heights), 'max': max(hr_heights), 'mean': np.mean(hr_heights), 'std': np.std(hr_heights), 'median': np.median(hr_heights)},
            'count': len(hr_sizes),
            'unique_sizes': list(set(hr_sizes))[:20],
        }
    }


def analyze_pixel_stats(lr_pixels, hr_pixels):
    lr_all = np.concatenate([p.flatten().reshape(-1, 3) for p in lr_pixels], axis=0)
    hr_all = np.concatenate([p.flatten().reshape(-1, 3) for p in hr_pixels], axis=0)
    
    return {
        'lr': {
            'mean_per_channel': lr_all.mean(axis=0).tolist(),
            'std_per_channel': lr_all.std(axis=0).tolist(),
            'global_mean': float(lr_all.mean()),
            'global_std': float(lr_all.std()),
        },
        'hr': {
            'mean_per_channel': hr_all.mean(axis=0).tolist(),
            'std_per_channel': hr_all.std(axis=0).tolist(),
            'global_mean': float(hr_all.mean()),
            'global_std': float(hr_all.std()),
        }
    }


def print_results(size_stats, pixel_stats, track_count):
    print("=" * 60)
    print("IMAGE SIZE ANALYSIS")
    print("=" * 60)
    print(f"\nTotal tracks analyzed: {track_count}")
    
    print("\n--- Low Resolution (LR) Images ---")
    lr = size_stats['lr']
    print(f"  Total images: {lr['count']}")
    print(f"  Width:  min={lr['width']['min']}, max={lr['width']['max']}, mean={lr['width']['mean']:.1f}, std={lr['width']['std']:.1f}, median={lr['width']['median']}")
    print(f"  Height: min={lr['height']['min']}, max={lr['height']['max']}, mean={lr['height']['mean']:.1f}, std={lr['height']['std']:.1f}, median={lr['height']['median']}")
    print(f"  Aspect ratio (W/H): mean={lr['width']['mean']/lr['height']['mean']:.2f}")
    print(f"  Sample unique sizes: {lr['unique_sizes'][:10]}")
    
    print("\n--- High Resolution (HR) Images ---")
    hr = size_stats['hr']
    print(f"  Total images: {hr['count']}")
    print(f"  Width:  min={hr['width']['min']}, max={hr['width']['max']}, mean={hr['width']['mean']:.1f}, std={hr['width']['std']:.1f}, median={hr['width']['median']}")
    print(f"  Height: min={hr['height']['min']}, max={hr['height']['max']}, mean={hr['height']['mean']:.1f}, std={hr['height']['std']:.1f}, median={hr['height']['median']}")
    print(f"  Aspect ratio (W/H): mean={hr['width']['mean']/hr['height']['mean']:.2f}")
    print(f"  Sample unique sizes: {hr['unique_sizes'][:10]}")
    
    print("\n--- Resolution Comparison ---")
    scale_w = hr['width']['mean'] / lr['width']['mean']
    scale_h = hr['height']['mean'] / lr['height']['mean']
    print(f"  HR/LR scale factor: width={scale_w:.2f}x, height={scale_h:.2f}x")
    
    print("\n" + "=" * 60)
    print("PIXEL VALUE STATISTICS (Normalized 0-1)")
    print("=" * 60)
    
    print("\n--- Low Resolution (LR) ---")
    lr_pix = pixel_stats['lr']
    print(f"  Mean per channel (R,G,B): [{lr_pix['mean_per_channel'][0]:.4f}, {lr_pix['mean_per_channel'][1]:.4f}, {lr_pix['mean_per_channel'][2]:.4f}]")
    print(f"  Std per channel (R,G,B):  [{lr_pix['std_per_channel'][0]:.4f}, {lr_pix['std_per_channel'][1]:.4f}, {lr_pix['std_per_channel'][2]:.4f}]")
    print(f"  Global mean: {lr_pix['global_mean']:.4f}")
    print(f"  Global std: {lr_pix['global_std']:.4f}")
    
    print("\n--- High Resolution (HR) ---")
    hr_pix = pixel_stats['hr']
    print(f"  Mean per channel (R,G,B): [{hr_pix['mean_per_channel'][0]:.4f}, {hr_pix['mean_per_channel'][1]:.4f}, {hr_pix['mean_per_channel'][2]:.4f}]")
    print(f"  Std per channel (R,G,B):  [{hr_pix['std_per_channel'][0]:.4f}, {hr_pix['std_per_channel'][1]:.4f}, {hr_pix['std_per_channel'][2]:.4f}]")
    print(f"  Global mean: {hr_pix['global_mean']:.4f}")
    print(f"  Global std: {hr_pix['global_std']:.4f}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDED TRAINING SIZE")
    print("=" * 60)
    
    avg_aspect = lr['width']['mean'] / lr['height']['mean']
    print(f"\nAverage aspect ratio: {avg_aspect:.2f}")
    
    target_height = 64
    target_width = int(target_height * avg_aspect)
    target_width = max(1, round(target_width / 32) * 32)
    print(f"Recommended size for 64 height: ({target_height}, {target_width})")
    
    target_height = 128
    target_width = int(target_height * avg_aspect)
    target_width = max(1, round(target_width / 32) * 32)
    print(f"Recommended size for 128 height: ({target_height}, {target_width})")
    
    print(f"\nCurrent dataset config: (64, 192)")
    print(f"  Aspect ratio used: {192/64:.2f}")
    
    all_mean = (np.array(lr_pix['mean_per_channel']) + np.array(hr_pix['mean_per_channel'])) / 2
    all_std = (np.array(lr_pix['std_per_channel']) + np.array(hr_pix['std_per_channel'])) / 2
    print(f"\nDataset normalization (mean): {all_mean.tolist()}")
    print(f"Dataset normalization (std): {all_std.tolist()}")
    print(f"\nImageNet normalization (mean): [0.485, 0.456, 0.406]")
    print(f"ImageNet normalization (std): [0.229, 0.224, 0.225]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA for image sizes in license plate dataset")
    parser.add_argument('--data_dir', type=str, default='data/train', help="Path to training data")
    parser.add_argument('--sample_limit', type=int, default=None, help="Limit number of tracks to analyze")
    args = parser.parse_args()
    
    print(f"Analyzing dataset: {args.data_dir}")
    size_stats, pixel_stats, track_count = analyze_dataset(args.data_dir, args.sample_limit)
    print_results(size_stats, pixel_stats, track_count)