# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC licence,
# found in the LICENSE_CELLDINO file in the root directory of this source tree.

#!/usr/bin/env python3
import torch
import torchvision
from dinov2.hub.backbones import celldino_hpa_vitl16, celldino_cp_vits8
from functools import partial
from dinov2.eval.utils import ModelWithIntermediateLayers

DEVICE = "cuda:0"
SAMPLE_IMAGES_DIR = ""  # path to directory with cell images.
MODELS_DIR = ""  # path to directory with pretrained models.


class self_normalize(object):
    def __call__(self, x):
        x = x / 255
        m = x.mean((-2, -1), keepdim=True)
        s = x.std((-2, -1), unbiased=False, keepdim=True)
        x -= m
        x /= s + 1e-7
        return x


normalize = self_normalize()

# ---------------------- Example inference on HPA-FoV dataset --------------------------

# 1- Read one human protein atlas HPA-FoV image (4 channels)
img = torchvision.io.read_image(SAMPLE_IMAGES_DIR + "HPA_FoV_00070df0-bbc3-11e8-b2bc-ac1f6b6435d0.png")

# 2- Normalise image as it was done for training
img_hpa_fov = img.unsqueeze(0).to(device=DEVICE)
img_hpa_fov = normalize(img_hpa_fov)

# 3- Load model
cell_dino_model = celldino_hpa_vitl16(
    pretrained_path=MODELS_DIR + "celldino_hpa_fov.pth",
)
print(cell_dino_model)
cell_dino_model.to(device=DEVICE)
cell_dino_model.eval()

# 4- Inference
features = cell_dino_model(img_hpa_fov)
print(features)

# 5- [Optional] feature extractor as used for linear evaluation
autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)
model_with_interm_layers = ModelWithIntermediateLayers(cell_dino_model, 4, autocast_ctx)
features_with_interm_layers = model_with_interm_layers(img_hpa_fov)

# ---------------------- Example inference on cell painting data --------------------------

# 1- Read one cell painting image (5 channels)
img = torchvision.io.read_image(SAMPLE_IMAGES_DIR + "CP_BBBC036_24277_a06_1_976@140x149.png")
img5_channels = torch.zeros([1, 5, 160, 160])
for c in range(5):
    img5_channels[0, c] = img[0, :, 160 * c : 160 * (c + 1)]
img5_channels = img5_channels.to(device=DEVICE)

# 2- Normalise image as it was done for training
img5_channels = normalize(img5_channels)

# 3- Load model
cell_dino_model = celldino_cp_vits8(
    pretrained_path=MODELS_DIR + "celldino_cp.pth",
)
print(cell_dino_model)
cell_dino_model.to(device=DEVICE)
cell_dino_model.eval()

# 4- Inference
features = cell_dino_model(img5_channels)
print(features)

# ---------------------- Example inference on HPA single cell dataset --------------------------

# Read one human protein atlas HPA single cell image (4 channels)
img = torchvision.io.read_image(SAMPLE_IMAGES_DIR + "HPA_single_cell_00285ce4-bba0-11e8-b2b9-ac1f6b6435d0_15.png")

# 2- Normalise image as it was done for training
img_hpa = img.unsqueeze(0).to(device=DEVICE)
img_hpa = normalize(img_hpa)

# 3- Load model
cell_dino_model = celldino_hpa_vitl16(
    pretrained_path=MODELS_DIR + "celldino_hpa_sc.pth",
)
print(cell_dino_model)
cell_dino_model.to(device=DEVICE)
cell_dino_model.eval()

# 4- Inference
features = cell_dino_model(img_hpa)
print(features)

torch.save(features.cpu(), "sample_features_hpa.pt")
