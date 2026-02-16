import cv2
import numpy as np
import torch

def crop_and_warp(image, json_data, target_size=(64, 320)):
    """
    Uses 4 corner points to rectify (warp) the plate into a perfect rectangle,
    then resizes to target_size with letterboxing.
    
    Args:
        image: Loaded image (numpy array)
        json_data: Dictionary with 'top-left', 'top-right', 'bottom-right', 'bottom-left'
        target_size: (Height, Width) for final model input
    """
    
    # 1. Parse Coordinates
    # Ensure they are int and float32 for OpenCV
    tl = np.array(json_data['top-left'], dtype=np.float32)
    tr = np.array(json_data['top-right'], dtype=np.float32)
    br = np.array(json_data['bottom-right'], dtype=np.float32)
    bl = np.array(json_data['bottom-left'], dtype=np.float32)
    
    # 2. Calculate Width and Height of the Plate (Euclidean Distance)
    # Width is max of top-width and bottom-width
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))
    
    # Height is max of left-height and right-height
    height_left = np.linalg.norm(tl - bl)
    height_right = np.linalg.norm(tr - br)
    max_height = max(int(height_left), int(height_right))
    
    # 3. Define Destination Points (A flat rectangle)
    # Top-Left, Top-Right, Bottom-Right, Bottom-Left
    dst_pts = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)
    
    # 4. Compute Perspective Transform Matrix
    src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # 5. Apply Warp
    warped = cv2.warpPerspective(image, M, (max_width, max_height), 
                                 flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_REPLICATE)
    
    # At this point, `warped` is the License Plate extracted perfectly 
    # from the crop, with correct aspect ratio (e.g., 53x23).
    
    # 6. Resize to Model Target Size with Letterboxing
    target_h, target_w = target_size
    
    # Calculate scaling ratio to fit the warped plate into the target box
    scale = min(target_w / max_width, target_h / max_height)
    
    new_w = int(max_width * scale)
    new_h = int(max_height * scale)
    
    # Resize
    resized = cv2.resize(warped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Letterbox (Center padding)
    # Use a neutral background color (White 255 is usually safer for plates than Black 0, 
    # unless plates are yellow, but White handles white/black plates reasonably well)
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # 7. Convert to Tensor
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(canvas).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    
    return tensor

if __name__ == "__main__":
    # Example usage
    track_path = "data/train/Scenario-A/Brazilian/track_00001"
    # image_hr = cv2.imread(f"{track_path}/hr-00001.png")
    json_data = open(track_path + '/annotations.json' ,'r')
    