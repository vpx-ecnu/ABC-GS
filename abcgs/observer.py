# style_observer.py
from dataclasses import dataclass
from typing import Dict, Optional
from abc import ABC
from tqdm import tqdm
import torch

@dataclass
class TrainingMetrics:
    iteration: int
    phase: int
    losses: Dict[str, float]
    timing: float
    
    
class TrainingObserver(ABC):
    
    def on_iteration_start(self, iteration: int): ...
    
    def on_iteration_end(self, metrics: TrainingMetrics): ...
    
    def on_phase_changed(self, previous: int, current: int): ...
    
    def on_training_end(self): ...
    

class ProgressTracker(TrainingObserver):
    
    def __init__(self, trainer):
        
        self.bar_format = "{l_bar}{bar:50}{r_bar}"
        self.trainer = trainer
        self.phase_bars: Dict[int, tqdm] = {}
        self.current_phase: Optional[int] = None
        self.update_interval: int = 10

    def on_phase_changed(self, previous: int, current: int):
        if previous in self.phase_bars:
            self.phase_bars[previous].close()
        
        if current not in self.phase_bars:
            phase = self.trainer.phases[current]
            self.phase_bars[current] = tqdm(
                total=phase.end_iter - phase.begin_iter + 1,
                desc=f"{phase.name.title():<18}",
                bar_format=self.bar_format
            )

    def on_iteration_end(self, metrics: TrainingMetrics):
        
        if metrics.iteration % self.update_interval != 0:
            return 
        
        if metrics.phase in self.phase_bars:
            self.phase_bars[metrics.phase].update(self.update_interval)
            self.phase_bars[metrics.phase].set_postfix(metrics.losses)
            
    def on_training_end(self):
        
        for phase_bar in self.phase_bars.values():
            phase_bar.close()
            
            
class CheckpointSaver(TrainingObserver):
    
    def __init__(self, trainer):
        self.trainer = trainer

    def on_iteration_end(self, metrics: TrainingMetrics):
        
        if metrics.iteration in self.trainer.config.ckpt.checkpoint_iterations:                
            print("\n[ITER {}] Saving Checkpoint".format(metrics.iteration))
            torch.save(
                (self.trainer.gaussians.capture(), metrics.iteration),
                self.trainer.config.model.model_path + "/chkpnt" + str(metrics.iteration) + ".pth",
            )
        
        if metrics.iteration in self.trainer.config.ckpt.save_iterations:
            self._save_gaussians(metrics.iteration)
    
    def _save_gaussians(self, iteration):
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        self.trainer.scene.save(self.trainer.total_iterations, self.trainer.config.model.model_path)
        
    def on_training_end(self):
        self._save_gaussians(self.trainer.total_iterations)
