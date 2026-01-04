import sys
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torchvision.transforms as T
from PIL import Image

# Add current directory to path so we can import depth_anything_v2
sys.path.append(os.getcwd())

from depth_anything_v2.dpt import DepthAnythingV2

def visualize_pca_advanced(image_path, model_path, output_path="pca_advanced.png"):
    # Load Image
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return
    
    # We use PIL to match run_pca_new.py transforms expectations
    raw_image = Image.open(image_path).convert('RGB')
    
    # Load Model
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
    
    # Preprocessing
    # We want to use the transforms from run_pca_new.py roughly
    # BUT we need to handle dynamic size.
    # run_pca_new.py hardcodes 40x40 patch count (560x560 px).
    # We should probably resize image to be divisible by 14 and reasonably sized.
    # Or just use the model's preprocessing but add blur?
    # run_pca_new.py used: GaussianBlur, Resize, CenterCrop.
    # Since we want to visualize the whole image, CenterCrop might lose content if aspect ratio differs.
    # Let's resize such that dimensions are multiples of 14.
    
    # To avoid OOM, let's limit max dimension to DINOv2 default 518 or slightly larger.
    # Vit-L is heavy. 6GB VRAM is tight for 1920x1080.
    max_dim = 518 * 2 # try 1036? or maybe just 518 to be safe and match original.
    # Original run_pca.py used 518. Let's stick to something safe.
    max_dim = 700 
    
    w, h = raw_image.size
    scale = min(max_dim / w, max_dim / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Round to nearest multiple of 14
    new_w = (new_w // 14) * 14
    new_h = (new_h // 14) * 14
    
    # It seems better to just resize to a fixed large size or keep original aspect ratio. 
    # Let's stick to simple resize to multiple of 14.
    
    patch_h = new_h // 14
    patch_w = new_w // 14
    
    print(f"Target size: {new_h}x{new_w} (Patches: {patch_h}x{patch_w})")
    
    transform = T.Compose([
        T.GaussianBlur(9, sigma=(0.1, 2.0)),
        T.Resize((new_h, new_w)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    input_tensor = transform(raw_image).unsqueeze(0).to(device)
    
    # Forward pass on backbone
    with torch.no_grad():
        features_dict = model.pretrained.forward_features(input_tensor)
        patch_tokens = features_dict["x_norm_patchtokens"] # (B, N, C)
    
    # Flatten features
    # (B, N, C) -> (N, C)
    feat_dim = patch_tokens.shape[-1]
    features = patch_tokens[0] # (N, C)
    features = features.cpu().numpy()
    
    print(f"Features shape: {features.shape}")
    
    # 1. PCA for feature inferred
    pca = PCA(n_components=3)
    pca.fit(features)
    pca_features = pca.transform(features)
    
    # Visualize PCA components for finding threshold (Optional, saving plot)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.hist(pca_features[:, 0], bins=50)
    plt.title("PC1")
    plt.subplot(1, 3, 2)
    plt.hist(pca_features[:, 1], bins=50)
    plt.title("PC2")
    plt.subplot(1, 3, 3)
    plt.hist(pca_features[:, 2], bins=50)
    plt.title("PC3")
    plt.savefig("pca_histograms.png")
    plt.close()
    
    # 2. Normalize features for visualization
    pca_features_rgb = pca_features.copy()
    for i in range(3):
        # Min-Max scaler to 0-1
        pca_features_rgb[:, i] = (pca_features_rgb[:, i] - pca_features_rgb[:, i].min()) / (pca_features_rgb[:, i].max() - pca_features_rgb[:, i].min())

    # Reshape
    pca_img = pca_features_rgb.reshape(patch_h, patch_w, 3)
    
    # Resize to original image size
    # Run_pca_new.py showed it as subplots. We want to save a high res image.
    orig_w, orig_h = raw_image.size
    pca_img_resized = cv2.resize(pca_img, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Clip to 0-1
    pca_img_resized = np.clip(pca_img_resized, 0, 1)

    # Plotting
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(raw_image)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(pca_img_resized)
    plt.title("Advanced PCA")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    img_path = "assets/frame_015314.jpg"
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        
    ckpt_path = "checkpoints/depth_anything_v2_vitl.pth"
    if len(sys.argv) > 2:
        ckpt_path = sys.argv[2]
        
    visualize_pca_advanced(img_path, ckpt_path)
