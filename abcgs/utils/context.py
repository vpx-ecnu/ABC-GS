
import torch 
from dataclasses import dataclass
from typing import Optional

@dataclass 
class StyleContext:
    
    image_width: int = None
    image_height: int = None
    
    depth_images: Optional[torch.Tensor] = None # [N, C, H, W]
    scene_images: Optional[torch.Tensor] = None  # [N, C, H, W]
    style_pixels_list: list[torch.Tensor] = None # list of Tensor [C, N_pixles]
    
    scene_masks: Optional[torch.Tensor] = None # [N, H, W]
    scene_features_list: list[torch.Tensor] = None # N list of Tensor [C, N_features]
    scene_features_masks: Optional[torch.Tensor] = None # [N, C, newH, newW]
    style_features_list: list[torch.Tensor] = None # list of Tensor [C, N_features]
    