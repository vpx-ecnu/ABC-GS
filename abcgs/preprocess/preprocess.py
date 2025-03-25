from .scene import _init_scene_images
from .scene import _init_scene_masks
from .style import _init_style_images
from .style import _init_style_masks
from .style import _erode
from .style import _isolate
from .segmenter import Segmenter
from ..utils.context import StyleContext
from ..utils.feature import FeatureExtractor
from ..utils.feature import merge
from ..utils.feature import get_separated_list
from icecream import ic
import torch 

@torch.no_grad
def preprocess(trainer):
    
    scene_images, depth_images = _init_scene_images(trainer)
    
    style_images_list = _init_style_images(trainer)
    
    scene_classes = trainer.config.style.scene_classes
    style_classes = trainer.config.style.style_classes
    
    segmenter = None if trainer.config.style.exec_mode == 'single' else Segmenter() 
    scene_masks = _init_scene_masks(trainer.config, scene_images, segmenter)
    style_masks_list = _init_style_masks(trainer.config, style_images_list, segmenter)
    
    if trainer.config.style.enable_erode:
        style_masks_list = _erode(style_masks_list, style_classes)
    if trainer.config.style.enable_isolate:
        style_images_list, style_masks_list = _isolate(style_images_list, style_masks_list, style_classes)
        
        # torch.save(style_masks_list, "./debug/isolate/masks.pt")
        # exit(0)
    
    trainer.feature_extractor = FeatureExtractor()
    
    scene_features_list, scene_features_masks_list = trainer.feature_extractor(scene_images, 
                                                                               scene_masks, 
                                                                               scene_classes)
    scene_features_masks = torch.stack(scene_features_masks_list)
    
    style_features_list, _ = trainer.feature_extractor(style_images_list,
                                                       style_masks_list, 
                                                       style_classes, 
                                                       downscale=False)
    
        
    trainer.ctx = StyleContext()
    trainer.ctx.scene_images = scene_images
    trainer.ctx.depth_images = depth_images
    
    trainer.ctx.scene_masks = scene_masks
    trainer.ctx.scene_features_list = scene_features_list
    trainer.ctx.scene_features_masks = scene_features_masks
    
    # Style Features are independent of Camera Pose, so merge them (for stylize loss)
    trainer.ctx.style_features_list = merge(style_features_list, style_classes)
    # Style Pixels are independent of Camera Pose, so merge them (for color transfer)
    style_pixels_list = []
    for i, style_image in enumerate(style_images_list):
        style_pixels_list.append(get_separated_list(style_image, style_masks_list[i], style_classes))
    trainer.ctx.style_pixels_list = merge(style_pixels_list, style_classes)
    
    
    
    
    
    
        