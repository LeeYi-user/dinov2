import sys
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import types

# Add current directory to path
sys.path.append(os.getcwd())

from depth_anything_v2.dinov2 import DINOv2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

# Global storage for attention weights
attention_weights = {}

def get_attention_forward(layer_idx):
    def forward(self, x, attn_bias=None):
        # Determine strict or not? 
        # Replicating the vanilla Attention.forward logic from dinov2_layers/attention.py
        # Ignoring attn_bias for standard inference if it's not supported by vanilla
        
        B, N, C = x.shape
        # qkv: (B, N, 3*dim) -> (B, N, 3, Heads, HeadDim) -> (3, B, Heads, N, HeadDim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        # q, k, v: (B, Heads, N, HeadDim)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        
        # attn: (B, Heads, N, N)
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        
        # Store attention weights (detached to save memory)
        attention_weights[layer_idx] = attn.detach().cpu()
        
        attn = self.attn_drop(attn)
        
        # Weighted sum: (B, Heads, N, HeadDim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    return forward

def load_and_patch_model(model_name='vitl'):
    print(f"Loading {model_name}...")
    model = DINOv2(model_name=model_name)
    
    checkpoint_path = r'checkpoints/depth_anything_v2_vitl.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('pretrained.'):
            new_key = key.replace('pretrained.', '')
            new_state_dict[new_key] = value
            
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Model loaded: {msg}")
    
    # Patch the attention blocks
    # blocks is a nn.ModuleList
    print("Patching attention layers...")
    for i, blk in enumerate(model.blocks):
        # blk.attn is the MemEffAttention module
        # We replace its forward method with our bound method
        blk.attn.forward = types.MethodType(get_attention_forward(i), blk.attn)
        model.blocks[i] = blk # Explicit assignment (though object is mutable)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    return model, device

def main():
    model, device = load_and_patch_model()
    
    input_dir = 'assets'
    output_dir = 'output/attention_vis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found in assets/")
        return
        
    # Process first image for demo
    filename = image_files[0]
    img_path = os.path.join(input_dir, filename)
    print(f"Processing {img_path}")
    
    # Preprocessing
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
    
    raw_image = cv2.imread(img_path)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    h_raw, w_raw, _ = raw_image.shape
    
    sample = transform({'image': raw_image})
    image_tensor = torch.from_numpy(sample['image']).unsqueeze(0).to(device)
    
    # Get patch grid dimensions
    _, h_new, w_new = sample['image'].shape
    h_patches = h_new // 14
    w_patches = w_new // 14
    print(f"Patch grid: {h_patches}x{w_patches}")
    
    # Clear previous weights
    attention_weights.clear()
    
    # Forward pass
    with torch.no_grad():
        model.forward_features(image_tensor)
        
    print(f"Captured attention from {len(attention_weights)} layers.")
    
    # Visualization
    num_layers = len(model.blocks)
    
    # Reload raw image for clean plotting
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    print(f"Visualizing all {num_layers} layers with individual heads...")
    
    for layer_idx in range(num_layers):
        if layer_idx not in attention_weights:
            continue
            
        attn = attention_weights[layer_idx] # (1, Heads, N, N)
        # attn shape: (1, 16, N_tokens, N_tokens) for ViT-L
        num_heads = attn.shape[1]
        
        # Calculate grid size for heads
        grid_size = int(np.ceil(np.sqrt(num_heads)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        fig.suptitle(f"Layer {layer_idx} Attention Heads", fontsize=24)
        
        axes_flat = axes.flatten()
        
        for head_idx in range(num_heads):
            ax = axes_flat[head_idx]
            
            # Extract attention for this head
            attn_head = attn[0, head_idx, 0, 1:] # CLS token (0) attending to patches (1:)
            
             # Check shape
            if attn_head.shape[0] != h_patches * w_patches:
                print(f"Warning: token count mismatch in layer {layer_idx} head {head_idx}")
                continue
                
            attn_map = attn_head.reshape(h_patches, w_patches).numpy()
            
            # Resize
            attn_resized = cv2.resize(attn_map, (w_raw, h_raw), interpolation=cv2.INTER_CUBIC)
            
            # Normalize
            attn_min, attn_max = attn_resized.min(), attn_resized.max()
            if attn_max > attn_min:
                attn_norm = (attn_resized - attn_min) / (attn_max - attn_min)
            else:
                attn_norm = attn_resized # Uniform attention
                
            # Heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * attn_norm), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Overlay
            alpha = 0.5
            overlay = (orig_img * (1 - alpha) + heatmap * alpha).astype(np.uint8)
            
            ax.imshow(overlay)
            ax.set_title(f"Head {head_idx}", fontsize=14)
            ax.axis('off')
            
        # Hide unused subplots
        for i in range(num_heads, len(axes_flat)):
            axes_flat[i].axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        savepath = os.path.join(output_dir, f"layer_{layer_idx:02d}_heads.png")
        plt.savefig(savepath)
        plt.close(fig)
        print(f"Saved {savepath}")
        
    print("Done.")

if __name__ == '__main__':
    main()
