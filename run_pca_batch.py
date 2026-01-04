
import sys
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision.transforms import Compose

# Add current directory to path to allow importing depth_anything_v2
sys.path.append(os.getcwd())

from depth_anything_v2.dinov2 import DINOv2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

def load_model():
    print("Loading model...")
    model = DINOv2(model_name='vitl')
    
    checkpoint_path = r'checkpoints/depth_anything_v2_vitl.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract pretrained DINOv2 weights
    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('pretrained.'):
            new_key = key.replace('pretrained.', '')
            new_state_dict[new_key] = value
            
    # Load into model
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Model loaded with message: {msg}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    return model, device

def main():
    model, device = load_model()
    
    input_dir = 'assets'
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images.")
    
    # --- Preprocessing Pipeline (same as dpt.py) ---
    input_size = 518
    transform = Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    all_patch_tokens = []
    metadata = [] # stores (filename, h_patches, w_patches)
    
    print("Extracting features...")
    with torch.no_grad():
        for filename in image_files:
            img_path = os.path.join(input_dir, filename)
            
            # Read image
            raw_image = cv2.imread(img_path)
            if raw_image is None:
                print(f"Failed to read {img_path}")
                continue
            
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
            h_raw, w_raw, _ = raw_image.shape
            
            # Apply transform
            sample = transform({'image': raw_image})
            image_tensor = torch.from_numpy(sample['image']).unsqueeze(0).to(device)
            
            # Calculate patch grid size
            # The transformed image has shape (C, H, W)
            _, h_new, w_new = sample['image'].shape
            h_patches = h_new // 14
            w_patches = w_new // 14
            
            # Forward pass
            features_dict = model.forward_features(image_tensor)
            patch_tokens = features_dict['x_norm_patchtokens'] # (1, N_patches, 1024)
            
            # Store
            all_patch_tokens.append(patch_tokens.squeeze(0).cpu())
            metadata.append({
                'filename': filename,
                'h_patches': h_patches,
                'w_patches': w_patches
            })
            
    if not all_patch_tokens:
        print("No features extracted.")
        return

    # Concatenate all tokens for global PCA
    print("Fitting PCA...")
    total_features = torch.cat(all_patch_tokens, dim=0) # (Total_patches, 1024)
    
    # PCA
    pca = PCA(n_components=1)
    # Convert to numpy (if not already cpu tensor)
    if isinstance(total_features, torch.Tensor):
        total_features_np = total_features.numpy()
    else:
        total_features_np = total_features
        
    pca.fit(total_features_np)
    pca_features = pca.transform(total_features_np)
    pca_features = pca_features[:, 0]
    
    # Min-Max normalize to 0-1
    pca_features = (pca_features - pca_features.min()) / \
                         (pca_features.max() - pca_features.min())

    # Reconstruct images
    print("Saving results...")
    current_idx = 0
    for meta in metadata:
        n_patches = meta['h_patches'] * meta['w_patches']
        
        # Slice features for this image
        img_pca = pca_features[current_idx : current_idx + n_patches]
        current_idx += n_patches
        
        # Reshape to (H_patches, W_patches)
        img_pca = img_pca.reshape(meta['h_patches'], meta['w_patches'])
        
        # Save
        out_name = os.path.splitext(meta['filename'])[0] + '_pca.png'
        out_path = os.path.join(output_dir, out_name)
        
        plt.imsave(out_path, img_pca)
        print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
