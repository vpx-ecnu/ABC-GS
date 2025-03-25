import torch 
import os
from tqdm import tqdm

def _init_scene_images(trainer):
    
    viewpoint_stack = trainer.scene.getTrainCameras()
    
    # colmap maybe change image's size
    min_h, min_w = 10000, 10000
    for i, view in enumerate(viewpoint_stack):
        min_h = min(min_h, view.image_height)
        min_w = min(min_w, view.image_width)
        
    depth_images = torch.zeros((len(viewpoint_stack), min_h, min_w), device=trainer.device)
    scene_images = torch.zeros((len(viewpoint_stack), 3, min_h, min_w), device=trainer.device)
    
    for i, view in enumerate(viewpoint_stack):
        depth_image = trainer.get_render_pkgs(view)["depth"]
        depth_images[i] = depth_image.squeeze().detach()
        scene_images[i] = view.original_image[:, :min_h, :min_w]
    
    return scene_images, depth_images

def _init_scene_masks(config, scene_images, segmenter):
    
    n, _, h, w = scene_images.shape
    scene_masks = torch.zeros((n, h, w), device=config.model.data_device)
    
    if config.style.exec_mode != 'single':
        
        scene_name = os.path.basename(config.model.model_path)
        cache_path = os.path.join(config.style.scene_segmentation_cache_path, 
                                f"{scene_name}/{config.style.scene_prompt}")
        masks_path = os.path.join(cache_path, "masks.pt")
        
        if os.path.exists(masks_path):
            scene_masks = torch.load(masks_path).to(device=config.model.data_device)
        else:
            os.makedirs(os.path.dirname(masks_path), exist_ok=True)

            for i, image in tqdm(enumerate(scene_images)):
                for j, label in enumerate(config.style.scene_prompt):
                    threshold = config.style.segmentation_threshold
                    scene_masks[i, segmenter(image, label) > threshold] = j + 1
            
            torch.save(scene_masks, masks_path)
        
    return scene_masks
    

