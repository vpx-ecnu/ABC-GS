from .base_phase import BasePhase
from gs.utils.loss_utils import l1_loss, ssim
from ..utils.color_transfer import color_transfer
from ..utils.image import render_ctx
from ..utils.image import render_viewpoint
import torch

class PreProcessPhase(BasePhase):
    
    def setup_phase(self):
        if self.trainer.config.style.enable_color_transfer:
            color_transfer(self.trainer.ctx, self.trainer.config)
        # render_ctx(self.trainer.ctx)
        # render_viewpoint(self.trainer, "./debug/bef")
        
    def cleanup_phase(self):
        pass
        # render_viewpoint(self.trainer, "./debug/aft")
        # exit(0)
    
    def process_iteration(self, iteration):
        
        viewpoint_cam = self._get_viewpoint_cam()
        
        with self.trainer.timer as timer:
            
            self.render_pkg = self.trainer.get_render_pkgs(viewpoint_cam)
            render_image = self.render_pkg["render"]
            scene_image = self.trainer.ctx.scene_images[viewpoint_cam.uid]
            
            Ll1 = l1_loss(render_image, scene_image)
            ssim_val = ssim(render_image, scene_image)
            
            loss = (
                (1.0 - self.trainer.config.opt.lambda_dssim) * Ll1 + 
                self.trainer.config.opt.lambda_dssim * (1.0 - ssim_val)
            )
            self.update(iteration, loss)
              
        return {
            "Points": f"{self.trainer.gaussians._opacity.shape[0]}",
            "Loss": f"{loss.item():.{7}f}"
        }, timer.elapsed_ms
        
    def _densification(self, iteration):
        delta_iteration = iteration - self.begin_iter
        
        gaussians = self.trainer.gaussians
        opt = self.trainer.config.opt
        scene = self.trainer.scene
        dataset = self.trainer.config.model
        
        viewspace_point_tensor = self.render_pkg["viewspace_points"]
        visibility_filter = self.render_pkg["visibility_filter"]
        radii = self.render_pkg["radii"]
        
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if delta_iteration == 0:
            return
        
        if delta_iteration % self.trainer.config.style.style_densification_interval == 0:
            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, 20, radii)
        
        if delta_iteration % opt.opacity_reset_interval == 0:
            gaussians.reset_opacity()
