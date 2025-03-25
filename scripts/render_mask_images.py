import sys
sys.path.append("./gs")
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)
sys.path.append(workspace_dir)

from abcgs.utils.image import render_depth_or_mask_images
import torch

pt_path = sys.argv[1]
masks = torch.load(os.path.join(pt_path, "masks.pt"))
for i in range(len(masks)):
    render_depth_or_mask_images(os.path.join(pt_path, f"{i}.jpg"), masks[i])
