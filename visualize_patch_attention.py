import sys
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import types
import argparse

# Add current directory to path
sys.path.append(os.getcwd())

from depth_anything_v2.dinov2 import DINOv2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

# Global storage for attention weights
attention_weights = {}

def get_attention_forward(layer_idx):
    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        # qkv: (B, N, 3*dim) -> (B, N, 3, Heads, HeadDim) -> (3, B, Heads, N, HeadDim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        # q, k, v: (B, Heads, N, HeadDim)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        
        # attn: (B, Heads, N, N)
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        
        # Store attention weights (detached to save memory)
        # Only store if it's the target layer (to save memory) - But user might want to see depth evolution
        # Storing all for now since batch size is 1
        attention_weights[layer_idx] = attn.detach().cpu()
        
        attn = self.attn_drop(attn)
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
    
    print("Patching attention layers...")
    for i, blk in enumerate(model.blocks):
        blk.attn.forward = types.MethodType(get_attention_forward(i), blk.attn)
        model.blocks[i] = blk
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    return model, device

def main():
    parser = argparse.ArgumentParser(description="Visualize DINOv2 Patch Attention")
    parser.add_argument('--image', type=str, default=None, help='Path to input image')
    args = parser.parse_args()

    # Interactive Mode
    default_image = 'assets/frame_015314.jpg'
    image_path = args.image if args.image else default_image
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    model, device = load_and_patch_model()
    
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
    
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        print("Failed to load image")
        return
        
    raw_image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    normalized_image_float = raw_image_rgb / 255.0
    h_raw, w_raw, _ = raw_image.shape
    
    # Prepare Tensor
    sample = transform({'image': normalized_image_float})
    image_tensor = torch.from_numpy(sample['image']).unsqueeze(0).to(device)
    
    # Get dimensions for mapping
    h_new, w_new = sample['image'].shape[1], sample['image'].shape[2]
    patch_size = 14
    h_patches = h_new // patch_size
    w_patches = w_new // patch_size
    
    scale_h = h_new / h_raw
    scale_w = w_new / w_raw
    
    print("Running forward pass (one-time)...")
    attention_weights.clear()
    with torch.no_grad():
        model.forward_features(image_tensor)
    print("Ready! Click on the image to visualize attention.")
    
    # State for interaction
    window_name = "DINOv2 Attention (Click to update)"
    current_layer = 23 # Default last layer
    
    display_image = raw_image.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal display_image
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked: ({x}, {y})")
            
            # Map coords
            x_new = int(x * scale_w)
            y_new = int(y * scale_h)
            
            patch_x = min(x_new // patch_size, w_patches - 1)
            patch_y = min(y_new // patch_size, h_patches - 1)
            
            # Clamp
            patch_x = max(0, patch_x)
            patch_y = max(0, patch_y)
            
            query_token_idx = 1 + patch_y * w_patches + patch_x
            
            # Get Attention
            attn = attention_weights[current_layer] # (1, Heads, N, N)
            
            # Mean Head
            attn_avg = attn.mean(dim=1)
            attn_patch = attn_avg[0, query_token_idx, 1:]
            
            attn_map = attn_patch.reshape(h_patches, w_patches).numpy()
            
            # Resize
            attn_resized = cv2.resize(attn_map, (w_raw, h_raw), interpolation=cv2.INTER_CUBIC)
            attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min())
            
            heatmap = cv2.applyColorMap(np.uint8(255 * attn_norm), cv2.COLORMAP_JET)
            # heatmap is BGR for OpenCV
            
            alpha = 0.6
            # Blend
            overlay = cv2.addWeighted(raw_image, 1 - alpha, heatmap, alpha, 0)
            
            # Marker
            cv2.drawMarker(overlay, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.putText(overlay, f"Layer {current_layer} | Pt: ({x},{y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            display_image = overlay
            cv2.imshow(window_name, display_image)
            
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    cv2.imshow(window_name, display_image)
    
    print("Controls:")
    print("  Click: Visualization Attention of that point")
    print("  'q':   Quit")
    print("  's':   Save current view")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            out_name = "output/interactive_save.png"
            cv2.imwrite(out_name, display_image)
            print(f"Saved {out_name}")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
