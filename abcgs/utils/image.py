import cv2 
import torch 
import numpy as np
import os

from gs.gaussian_renderer import render

def read_and_resize_image(image_path, target_height):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scene_height, scene_width = image_rgb.shape[:2]
    scale_ratio = target_height / scene_height
    target_width = int(scene_width * scale_ratio)
    
    resized_image = cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
    resized_image = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
    return resized_image

def labels_downscale(labels, new_dim):
    """
    Downscales the labels to a new dimension.

    @param labels: Tensor of labels. Shape: [H, W]
    @param new_dim: Tuple of new dimensions (NH, NW)
    @return: Downscaled labels
    """
    H, W = labels.shape
    NH, NW = new_dim
    r_indices = torch.linspace(0, H-1, NH).long()
    c_indices = torch.linspace(0, W-1, NW).long()
    return labels[r_indices[:, None], c_indices]

def render_depth_or_mask_images(path, image):
    
    image = image.detach().cpu().numpy().squeeze()
    depth_map_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = np.uint8(depth_map_normalized)
    # depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(path, depth_map_normalized)
    
def render_RGBcolor_images(path, image):
    
    image = image.detach().permute(1, 2, 0).clamp(min=0.0, max=1.0).cpu().numpy()
    image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)
    
def render_ctx(ctx, path="./debug"):
    
    depth_path = os.path.join(path, "depth/")
    scene_path = os.path.join(path, "scene/")
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(scene_path, exist_ok=True)
    
    for i in range(ctx.scene_images.shape[0]):
        render_RGBcolor_images(os.path.join(scene_path, f"{int(i):04d}.png"), ctx.scene_images[i])
    for i in range(ctx.depth_images.shape[0]):
        render_depth_or_mask_images(os.path.join(depth_path, f"{int(i):04d}.png"), ctx.depth_images[i])
        
        

def render_viewpoint(trainer, path="./debug"):
    
    render_path = os.path.join(path, "render/")
    os.makedirs(render_path, exist_ok=True)
    
    for i, view in enumerate(trainer.scene.getTrainCameras()):
        render_image = trainer.get_render_pkgs(view)["render"]
        cur_render_path = os.path.join(render_path, f"{int(i):04d}.png")
        render_RGBcolor_images(cur_render_path, render_image)