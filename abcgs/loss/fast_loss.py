from .base_loss import StylizeLoss
import torch
import torch.nn.functional as F


class FASTLoss(StylizeLoss):
    
    @torch.no_grad
    def cal_p(self, cf, sf):
        cf_size = cf.size()
        sf_size = sf.size()
        
        k_cross = 5

        cf_temp = cf
        sf_temp = sf

        cf_n = F.normalize(cf, 2, 0)
        sf_n = F.normalize(sf, 2, 0)
        
        dist = torch.mm(cf_n.t(), sf_n)  # inner product,the larger the value, the more similar

        hcwc, hsws = cf_size[1], sf_size[1]
        U = torch.zeros(hcwc, hsws).type_as(cf_n).to(self.config.model.data_device)  # construct affinity matrix "(h*w)*(h*w)"

        index = torch.topk(dist, k_cross, 0)[1]  # find indices k nearest neighbors along row dimension
        value = torch.ones(k_cross, hsws).type_as(cf_n).to(self.config.model.data_device) # "KCross*(h*w)"
        U.scatter_(0, index, value)  # set weight matrix

        index = torch.topk(dist, k_cross, 1)[1]  # find indices k nearest neighbors along col dimension
        value = torch.ones(hcwc, k_cross).type_as(cf_n).to(self.config.model.data_device)
        U.scatter_(1, index, value)  # set weight matrix
        
        n_cs = torch.sum(U)
        U = U / n_cs
        D1 = torch.diag(torch.sum(U, dim=1)).type_as(cf).to(self.config.model.data_device)
        
        A = torch.mm(torch.mm(cf_temp, D1), cf_temp.t())
        regularization_term = torch.eye(A.size()[0]).type_as(A).to(self.config.model.data_device) * 1e-12
        A += regularization_term
        B = torch.mm(torch.mm(cf_temp, U), sf_temp.t())
        
        try:
            p = torch.linalg.solve(A, B)
        except Exception as e:
            print(e)
            p = torch.eye(cf_size[0]).type_as(cf).to(self.config.model.data_device)
        return p


    @torch.no_grad
    def transform(self, render_feats, style_feats):
        p = self.cal_p(render_feats, style_feats)
        return torch.mm(p.t(), render_feats)
    
    
    def __call__(self, render_feats_list, style_feats_list):
        
        def cos_loss(a, b):
            cossim = F.cosine_similarity(a, b, dim=1)
            return (1.0 - cossim).mean()
        
        stylize_loss = 0
        for i in range(self.config.style.scene_classes):
            a = render_feats_list[i]
            b = self.transform(a, style_feats_list[self.config.style.override_matches[i]])
            stylize_loss += cos_loss(a, b)
                
        return stylize_loss