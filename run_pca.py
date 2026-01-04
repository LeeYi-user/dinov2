import sys
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Add current directory to path so we can import depth_anything_v2
sys.path.append(os.getcwd())

from depth_anything_v2.dpt import DepthAnythingV2

def visualize_pca(image_path, model_path, output_path="pca_vis.png"):
    # Load Image
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return
    
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Load Model
    # Encoder is 'vitl' as per user request
    # DINOv2 in DepthAnythingV2 uses features=256, out_channels=[256, 512, 1024, 1024] by default in standard configs often?
    # Let's check dpt.py defaults. They are explicit in constructor.
    # We should match what was used for the checkpoint.
    # The checkpoint name is depth_anything_v2_vitl.pth.
    # Usually this corresponds to the 'vitl' configuration.
    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    
    print(f"Loading checkpoint from {model_path}...")
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False) 
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    model.eval()
    
    # Get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Preprocess
    # image2tensor returns (1, C, H, W) and (orig_h, orig_w)
    # We use a fixed input size or let it resize. 518 is DINOv2 default.
    input_tensor, (orig_h, orig_w) = model.image2tensor(raw_image, input_size=518)
    input_tensor = input_tensor.to(device)
    
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Forward pass on backbone
    with torch.no_grad():
        # model.pretrained is the DINOv2 model
        # forward_features returns dict
        features_dict = model.pretrained.forward_features(input_tensor)
        patch_tokens = features_dict["x_norm_patchtokens"] # (B, N, C)
    
    print(f"Patch tokens shape: {patch_tokens.shape}")
    
    # PCA
    # Shape (B, N, C) -> (N, C) assuming B=1
    tokens = patch_tokens[0].cpu().numpy()
    
    # PCA
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(tokens)
    
    # Normalize to [0, 1] for RGB
    # Use robust min-max (ignoring extreme outliers can be better, but simple min-max is standard)
    pca_features = (pca_features - pca_features.min(0)) / (pca_features.max(0) - pca_features.min(0))
    
    # Reshape to grid
    h_in, w_in = input_tensor.shape[-2:]
    h_patch = h_in // 14
    w_patch = w_in // 14
    
    print(f"Grid size: {h_patch}x{w_patch} (Tokens: {tokens.shape[0]})")
    
    if h_patch * w_patch != tokens.shape[0]:
        print("Warning: token count mismatch")
    
    pca_img = pca_features.reshape(h_patch, w_patch, 3)
    
    # Resize to original image size
    # Using Nearest to see patches, or Cubic for smooth.
    # DINOv2 figures usually look somewhat smooth but patch structure is visible.
    # Let's use components to visualize.
    pca_img_resized = cv2.resize(pca_img, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Save raw PCA image for clarity
    pca_img_uint8 = (pca_img_resized * 255).astype(np.uint8)
    # Convert RGB to BGR for cv2 save if needed, but we use maplotlib
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(pca_img_resized)
    plt.title("DINOv2 PCA Visualization")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    # Ensure args
    img_path = "assets/frame_015314.jpg"
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        
    ckpt_path = "checkpoints/depth_anything_v2_vitl.pth"
    if len(sys.argv) > 2:
        ckpt_path = sys.argv[2]
        
    visualize_pca(img_path, ckpt_path)
