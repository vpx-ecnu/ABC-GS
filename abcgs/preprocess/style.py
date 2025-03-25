import torch
from ..utils.image import read_and_resize_image
import os
import tqdm
import numpy as np
import cv2
from icecream import ic

def _init_style_images(trainer):
    
    style_images = []
    for path in trainer.config.style.style_images:
        style_images.append(
            read_and_resize_image(
                path, trainer.config.style.downsampled_image_size,
            ).to(device=trainer.device).contiguous()
        )
    return style_images
    
def _init_style_masks(config, style_images, segmenter):
    
    num_style_images = len(style_images)
    
    style_masks = []
    for i in range(num_style_images):
        _, h, w = style_images[i].shape
        style_masks.append(torch.zeros((h, w), device=config.model.data_device))
    
    if config.style.exec_mode != 'sematic':
        for i in range(num_style_images):
            style_masks[i][:, :] = i
    else:
        # For Sematic Execute Mode
        style_name = os.path.basename(config.style.style_images[0])
        cache_path = os.path.join(config.style.style_segmentation_cache_path, 
                                f"{style_name}/{config.style.style_prompt}/")
        masks_path = os.path.join(cache_path, "masks.pt")
        
        if os.path.exists(masks_path):
            style_masks = torch.load(masks_path)
            for i in range(len(style_masks)):
                style_masks[i] = style_masks[i].to(device=config.model.data_device)
        else:
            os.makedirs(os.path.dirname(masks_path), exist_ok=True)
            for i, label in enumerate(config.style.style_prompt):
                threshold = config.style.segmentation_threshold
                style_masks[0][segmenter(style_images[0], label) > threshold] = i + 1

            torch.save(style_masks, masks_path)
            
    return style_masks


def _erode(masks, classes):
    
    erode_masks = [torch.full_like(masks[0], -1, device=masks[0].device)]
    
    for i in range(classes):
        
        original_mask = (masks[0] == i)
        
        new_mask = original_mask.cpu().numpy().astype(np.uint8) * 255
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        new_mask = cv2.erode(new_mask, kernel, iterations=1)
        new_mask = torch.tensor((new_mask / 255).astype(bool), device=masks[0].device)
        
        erode_masks[0][new_mask] = i 
        
    return erode_masks

def _isolate(images, masks, classes):
    
    c, h, w = images[0].shape
    
    isolate_masks_list = []
    isolate_images_list = []
    
    for i in range(classes):
        isolate_mask = torch.full((h, w), -1, device=images[0].device)
        isolate_mask[masks[0] == i] = i
        isolate_masks_list.append(isolate_mask)
        
        isolate_image = torch.zeros((c, h, w), device=images[0].device)
        isolate_image[:, masks[0] == i] = images[0][:, masks[0] == i]
        isolate_images_list.append(isolate_image)
        
    return isolate_images_list, isolate_masks_list
        