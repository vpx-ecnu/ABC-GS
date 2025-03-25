import torch
from icecream import ic
from .feature import merge

def color_transfer(ctx, config):
    
    def match_colors(scene_images, style_image):
        sh = scene_images.shape
        image_set = scene_images.view(-1, 3)
        style_img = style_image.view(-1, 3).to(image_set.device)

        mu_c = image_set.mean(0, keepdim=True)
        mu_s = style_img.mean(0, keepdim=True)

        cov_c = torch.matmul((image_set - mu_c).transpose(1, 0), image_set - mu_c).float() / float(image_set.size(0))
        cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s).float() / float(style_img.size(0))

        u_c, sig_c, _ = torch.svd(cov_c)
        u_s, sig_s, _ = torch.svd(cov_s)

        u_c_i = u_c.transpose(1, 0)
        u_s_i = u_s.transpose(1, 0)

        scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
        scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

        tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
        tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

        image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
        image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

        color_tf = torch.eye(4).float().to(tmp_mat.device)
        color_tf[:3, :3] = tmp_mat
        color_tf[:3, 3:4] = tmp_vec.T
        return image_set, color_tf
    
    
    nhwc_scene_images = ctx.scene_images.permute(0, 2, 3, 1)
    
    for i in range(config.style.scene_classes):
        scene_mask = (ctx.scene_masks == i) # [N, H, W]
        scene_pixels = nhwc_scene_images[scene_mask, :] # [N_pixels, C]
        style_pixels = ctx.style_pixels_list[config.style.override_matches[i]].permute(1, 0) # [N_pixels, C]

        color_transferred_scene_pixels, _ = match_colors(scene_pixels, style_pixels)
        nhwc_scene_images[scene_mask, :] = color_transferred_scene_pixels
        
    ctx.scene_images = nhwc_scene_images.permute(0, 3, 1, 2)