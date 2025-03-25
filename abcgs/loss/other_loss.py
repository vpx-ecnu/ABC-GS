import torch

def content_loss_fn(render_feats_list, scene_feats_list):
    content_loss = 0
    for (render_feat, scene_feat) in zip(render_feats_list, scene_feats_list):
        content_loss += torch.mean((render_feat - scene_feat) ** 2)
    return content_loss

def image_tv_loss_fn(render_image):
    image = render_image.unsqueeze(0)
    
    w_variance = torch.mean(torch.pow(image[:, :, :-1] - image[:, :, 1:], 2))
    h_variance = torch.mean(torch.pow(image[:, :-1, :] - image[:, 1:, :], 2))
    img_tv_loss = (h_variance + w_variance) / 2.0
    return img_tv_loss