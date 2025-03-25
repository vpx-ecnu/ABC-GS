from .base_loss import StylizeLoss
import torch
import torch.nn.functional as F


class KNNFMLoss(StylizeLoss):
    
    def __call__(self, render_feats_list, style_feats_list):
        
        def get_cos_matrix(a, b):
            a_tmp = F.normalize(a, dim=0)
            b_tmp = F.normalize(b, dim=0)
            return 1.0 - torch.matmul(a_tmp.T, b_tmp)
        
        stylize_loss = 0
        for i in range(self.config.style.scene_classes):
            a = render_feats_list[i]
            b = style_feats_list[self.config.style.override_matches[i]]
            dists = -get_cos_matrix(a, b)
            kmin_dist, _ = torch.topk(dists, k=5, dim=1)
            kmean_dist = torch.mean(kmin_dist, dim=1)
            stylize_loss += -torch.mean(kmean_dist)
            
        return stylize_loss