from abc import ABC, abstractmethod
from random import randint
from ..utils.image import render_RGBcolor_images

class BasePhase(ABC):
    
    def __init__(self, trainer, id, name, begin_iter, end_iter):
        self.trainer = trainer
        self.id = id
        self.name = name
        self.begin_iter = begin_iter
        self.end_iter = end_iter
        self.viewpoint_stack = None

    def update(self, iteration, loss):
        if self.name == "Stylize" or self.name == "Post Process":
            render_RGBcolor_images("./debug/image.jpg", self.render_pkg["render"])
        loss.backward()
        self._densification(iteration)
        self.trainer.gaussians.optimizer.step()
        self.trainer.gaussians.optimizer.zero_grad(set_to_none=True)
    
    def _get_viewpoint_cam(self):
        if not self.viewpoint_stack or len(self.viewpoint_stack) == 0:
            self.viewpoint_stack = self.trainer.scene.getTrainCameras().copy()
        return self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
    
    def setup_phase(self): ...
    def cleanup_phase(self): ...
    
    @abstractmethod
    def process_iteration(self, iteration): ...
    
    def _densification(self, iteration): ...
    