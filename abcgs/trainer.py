# style_trainer.py
from typing import Dict, List
import torch
from icecream import ic
from gs.scene import GaussianModel, Scene
from gs.gaussian_renderer import render

from .configs import ConfigManager
# from style_utils import CUDATimer
from .phases.pre_process_phase import PreProcessPhase
from .phases.stylize_phase import StylizePhase
from .phases.post_process_phase import PostProcessPhase
from .phases.pre_process_phase import *
from .phases.stylize_phase import *
from .phases.pre_process_phase import *
from .preprocess.preprocess import preprocess
from .utils.timer import CUDATimer
from .utils.network_gui import handle_network_gui
from .observer import TrainingMetrics
from .observer import TrainingObserver
from .observer import ProgressTracker
from .observer import CheckpointSaver
from .loss.fast_loss import FASTLoss
from .loss.nnfm_loss import NNFMLoss
from .loss.knnfm_loss import KNNFMLoss
from .loss.gram_loss import GRAMLoss

class StyleTrainer:
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.device = config.model.data_device
        self.timer = CUDATimer()
        
        stylize_loss_dict = {
            "fast": FASTLoss,
            "nnfm": NNFMLoss,
            "knnfm": KNNFMLoss,
            "gram": GRAMLoss
        }
        self.stylize_loss_fn = stylize_loss_dict[config.style.method](config)
        
        self._initialize_components()
        self._initialize_phases()
        self._initialize_observers()
        preprocess(self)
        
        
    def train(self):
        
        for self.iteration in range(1, self.total_iterations + 1):
            handle_network_gui(self)
            self._train_iteration()
            
        for observer in self.observers:
            observer.on_training_end()
    
        
    def _train_iteration(self):
        
        # Phase transition
        new_phase = self._determine_current_phase()
        if new_phase != self.cur_phase:
            self._transition_to_phase(new_phase)
        phase = self.phases[self.cur_phase]
        
        # Update Observer
        for observer in self.observers:
            observer.on_iteration_start(self.iteration)
        
        # Original Gaussian repo's operations
        self.gaussians.update_learning_rate(self.iteration)
        self.config.set_debug(True if self.iteration - 1 == self.config.app.debug_from else False)
        
        # Calculate loss
        losses, timing = phase.process_iteration(self.iteration)
        
        metrics = TrainingMetrics(
            iteration=self.iteration,
            phase=self.cur_phase,
            losses=losses,
            timing=timing
        )
        
        # Update Observer
        for observer in self.observers:
            observer.on_iteration_end(metrics)
    
    def _determine_current_phase(self):
        new_phase = self.cur_phase
        if self.cur_phase == -1:
            return 0
        while self.iteration > self.phases[new_phase].end_iter:
            new_phase += 1
        return new_phase  
    
    def _transition_to_phase(self, new_phase):
        
        if self.cur_phase != -1:
            self.phases[self.cur_phase].cleanup_phase()
            self.phases[self.cur_phase] = None
        
        self.cur_phase = new_phase
        self.phases[self.cur_phase].setup_phase()
        
        for observer in self.observers:
            observer.on_phase_changed(self.cur_phase, new_phase)    
            
    def _initialize_components(self):
        
        # Just as same as original gaussian repo's initialization
        self.gaussians = GaussianModel(self.config.model.sh_degree)
        # ic(self.config.model.model_path)
        self.scene = Scene(self.config.model, self.gaussians, -1, shuffle=False)
        
        bg_color = [1, 1, 1] if self.config.model.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.config.model.data_device)
            
        self.gaussians.training_setup(self.config.opt)
        
    def _initialize_phases(self):
        
        self.cur_phase = -1
        self.phases = []
        self.total_iterations = 0
        
        phase_id = 0
        
        def _add_phase(phase, phase_name, num_iter):
            if num_iter == 0:
                return
            
            nonlocal phase_id
            
            begin_iter = self.total_iterations + 1
            end_iter = self.total_iterations + num_iter
            
            self.phases.append(phase(self, phase_id, phase_name, begin_iter, end_iter))
            
            self.total_iterations += num_iter
            phase_id += 1
        
        _add_phase(PreProcessPhase,  "Pre Process", self.config.style.iterations_pre_process)
        _add_phase(StylizePhase,     "Stylize", self.config.style.iterations_stylize)
        _add_phase(PostProcessPhase, "Post Process", self.config.style.iterations_post_process)
        
        
    def _initialize_observers(self):
        
        self.observers: List[TrainingObserver] = [
            ProgressTracker(self),
            CheckpointSaver(self)
        ]

    def _get_background(self):
        if self.config.opt.random_background:
            return torch.rand((3), device=self.config.model.data_device)
        return self.background

    def get_render_pkgs(self, viewpoint_cam):
        return render(viewpoint_cam, self.gaussians, self.config.pipe, self._get_background())

