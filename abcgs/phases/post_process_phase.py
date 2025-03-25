from .base_phase import BasePhase
from gs.utils.loss_utils import l1_loss, ssim
from ..utils.color_transfer import color_transfer

class PostProcessPhase(BasePhase):
    
    def setup_phase(self):
        viewpoint_stack = self.trainer.scene.getTrainCameras()
        for i, view in enumerate(viewpoint_stack):
            pkg = self.trainer.get_render_pkgs(view)
            self.trainer.ctx.scene_images[i] = pkg["render"].detach()
        
        if self.trainer.config.style.enable_color_transfer:
            color_transfer(self.trainer.ctx, self.trainer.config)
        self.trainer.gaussians._scaling.requires_grad_(False)
        self.trainer.gaussians._xyz.requires_grad_(False)
        self.trainer.gaussians._opacity.requires_grad_(False)
    
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