from .base_loss import StylizeLoss
import torch
import torch.nn.functional as F


class GRAMLoss(StylizeLoss):
    
    def __call__(self, render_feats_list, style_feats_list):
        
        def gram_matrix(features, center=False):
            if center:
                features = features - features.mean(dim=-1, keepdims=True)
            return torch.mm(features, features.T)
        
        stylize_loss = 0
        for i in range(self.config.style.scene_classes):
            a = gram_matrix(render_feats_list[i])
            b = gram_matrix(style_feats_list[self.config.style.override_matches[i]])
            stylize_loss += torch.mean((a - b) ** 2)
        return stylize_loss