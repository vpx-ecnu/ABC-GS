from .base_phase import BasePhase
from icecream import ic
from ..loss.other_loss import content_loss_fn
from ..loss.other_loss import image_tv_loss_fn
import torch

class StylizePhase(BasePhase):
    
    def setup_phase(self):
        
        self.initial_opacity = self.trainer.gaussians._opacity.clone().detach()
        self.initial_scaling = self.trainer.gaussians._scaling.clone().detach()
    
    def process_iteration(self, iteration):
        
        viewpoint_cam = self._get_viewpoint_cam()
        
        with self.trainer.timer as timer:
            
            self.render_pkg = self.trainer.get_render_pkgs(viewpoint_cam)
            render_image = self.render_pkg["render"]
            
            render_feature_list, _ = self.trainer.feature_extractor(
                render_image.unsqueeze(0),
                self.trainer.ctx.scene_masks[viewpoint_cam.uid].unsqueeze(0),
                self.trainer.config.style.scene_classes)
            # Batch is 1, so get the first render_feature_list
            render_feature_list = render_feature_list[0]
            
            stylize_loss = self.trainer.stylize_loss_fn(render_feature_list, self.trainer.ctx.style_features_list)
            content_loss = content_loss_fn(render_feature_list, self.trainer.ctx.scene_features_list[viewpoint_cam.uid])
            image_tv_loss = image_tv_loss_fn(render_image)
            
            render_depth = self.render_pkg["depth"]
            scene_depth = self.trainer.ctx.depth_images[viewpoint_cam.uid]
            depth_loss = torch.mean((render_depth - scene_depth) ** 2)
            
            loss_delta_opacity = torch.norm(self.trainer.gaussians._opacity - self.initial_opacity)
            loss_delta_scaling = torch.norm(self.trainer.gaussians._scaling - self.initial_scaling)
            
            loss = (
                self.trainer.config.style.lambda_stylize * stylize_loss +
                self.trainer.config.style.lambda_content * content_loss +
                self.trainer.config.style.lambda_img_tv * image_tv_loss + 
                self.trainer.config.style.lambda_depth * depth_loss +
                self.trainer.config.style.lambda_delta_opacity * loss_delta_opacity +
                self.trainer.config.style.lambda_delta_scaling * loss_delta_scaling
            )
            
            self.update(iteration, loss)
            
        
        return {
            "Points": f"{self.trainer.gaussians._opacity.shape[0]}",
            "Loss": f"{loss.item():.{7}f}"
        }, timer.elapsed_ms