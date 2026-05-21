from typing import Any
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
from PIL import Image
import os
import sys
# Ensure ml-depth-pro is on sys.path before importing depth_pro
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "submodules", "ml-depth-pro", "src"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import depth_pro
depth_pro_model, depth_pro_transform = depth_pro.create_model_and_transforms(device=torch.device("cuda"))
depth_pro_model.eval()
for param in depth_pro_model.parameters():
    param.requires_grad = False
    
def estimate_depth_pro(tensor, mode='test'):
    if mode == 'test':
        with torch.no_grad():
            transformed_image = depth_pro_transform(tensor)
            prediction = depth_pro_model.infer(transformed_image)
            render_depth_pro = prediction["depth"]
            return render_depth_pro
    else:
        transformed_image = depth_pro_transform(tensor)
        prediction = depth_pro_model.infer(transformed_image)
        render_depth_pro = prediction["depth"]
        return render_depth_pro

def apply_colormap(depth_map, cmap_name='jet'):
    # Check input type and shape
    if isinstance(depth_map, torch.Tensor):
        if depth_map.dim() > 2:
            depth_map = depth_map.squeeze(0)  # Remove the batch dimension, resulting in (H, W)
        depth_np = depth_map.cpu().numpy()
    elif isinstance(depth_map, np.ndarray):
        if depth_map.ndim > 2:
            depth_map = np.squeeze(depth_map)
        depth_np = depth_map
    else:
        raise TypeError("Input must be either a torch.Tensor or a numpy.ndarray")

    # Apply colormap
    cmap = plt.get_cmap(cmap_name)
    colored_depth = cmap(depth_np)
    
    # Convert back to torch tensor (H, W, 4) -> (3, H, W)
    colored_depth_tensor = torch.from_numpy(colored_depth).permute(2, 0, 1)[:3]
    
    return colored_depth_tensor
