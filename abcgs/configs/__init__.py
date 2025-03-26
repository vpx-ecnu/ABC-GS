from dataclasses import dataclass
from simple_parsing import ArgumentParser, field, Serializable
from simple_parsing.helpers import list_field
from datetime import datetime
from zoneinfo import ZoneInfo
import os

@dataclass
class ModelConfig(Serializable):
    """
    Just as same as ModelParams
    """
    sh_degree: int = 3
    source_path: str = field(None, alias="-s")
    images: str = field("images", alias="-i")
    depths: str = field("", alias="-d")
    resolution: int = field(-1, alias="-r")
    white_background: bool = field(False, alias="-w", action="store_true")
    train_test_exp: bool = field(False, action="store_true")
    data_device: str = "cuda"
    eval: bool = field(False, action="store_true")
    model_path: str = field(None, alias="-m")


@dataclass
class PipelineConfig(Serializable):
    """
    Just as same as PipelineParams
    """
    convert_SHs_python: bool = field(False, action="store_true")
    compute_cov3D_python: bool = field(False, action="store_true")
    debug: bool = field(False, action="store_true")
    antialiasing: bool = field(False, action="store_true")
  

@dataclass
class OptimizationConfig(Serializable):
    """
    Just as same as OptimizationParams
    """
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    exposure_lr_init: float = 0.01
    exposure_lr_final: float = 0.001
    exposure_lr_delay_steps: float = 0
    exposure_lr_delay_mult: float = 0.0
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002
    depth_l1_weight_init: float = 1.0
    depth_l1_weight_final: float = 0.01
    random_background: bool = field(False, action="store_true")
    optimizer_type: str = "default"


@dataclass
class ApplicationConfig(Serializable):
    ip: str = "127.0.0.1"
    port: int = 6009
    debug_from: int = -1
    detect_anomaly: bool = field(False, action="store_true")
    quiet: bool = field(False, action="store_true")
    
    enable_wandb: bool = field(False, action="store_true")
    
@dataclass
class CheckpointConfig(Serializable):
    save_iterations: list[int] = list_field()
    checkpoint_iterations: list[int] = list_field()
    start_checkpoint: str = None

@dataclass
class StyleConfig(Serializable):
    # Specify which mode to use
    exec_mode: str = field("single", choices=["single", "sematic", "multi"])
    
    # Model path for stylization output
    stylized_model_path: str = field(None, alias="-o")
    # Densification interval for pre-process and post process
    style_densification_interval: int = 50
    # Densification threshold Interval for pre-process and post process
    style_densification_threshold: float = 0.00001
    
    style_images: list[str] = list_field()
    style_prompt: list[str] = list_field()
    scene_prompt: list[str] = list_field()
    
    # Path for lang_seg's output for style
    style_segmentation_cache_path: str = "./cache/style"
    # Path for lang_seg's output for scene
    scene_segmentation_cache_path: str = "./cache/scene"
    
    segmentation_threshold: float = 0.7
    
    # Loss method for training
    method: str = field("fast", choices=["fast", "nnfm", "knnfm", "gram"])
    
    enable_erode: bool = field(False, action="store_true")
    enable_isolate: bool = field(False, action="store_true")
    enable_color_transfer: bool = field(False, action="store_true")
    
    iterations_pre_process: int = 400
    iterations_stylize: int = 600
    iterations_post_process: int = 400
    
    lambda_stylize: float = 2
    lambda_content: float = 0.005
    lambda_img_tv: float = 0.02
    lambda_depth: float = 0.01
    lambda_delta_opacity: float = 1
    lambda_delta_scaling: float = 1
    
    # Image size for downsampled style images
    downsampled_image_size: int = 256
    
    # Matches for scene and style, e.g. override_matches[0] = 1, scene[0] <-> style[1]
    override_matches: list[int] = list_field()
    
    random_seed: int = 42
    
@dataclass
class ConfigManager(Serializable):
    
    model: ModelConfig
    opt: OptimizationConfig
    pipe: PipelineConfig
    app: ApplicationConfig
    style: StyleConfig
    ckpt: CheckpointConfig
    
    def __init__(self, raw_config):
        self.model = raw_config.model
        self.opt = raw_config.opt
        self.pipe = raw_config.pipe
        self.app = raw_config.app
        self.style = raw_config.style
        self.ckpt = raw_config.ckpt
        
        self._check_params()
        self._generate_output_path()
        self._save_args()
        
    def _check_params(self):
        
        for path in self.style.style_images:
            assert os.path.exists(path), f"{path} dost not exists."
            
        if self.style.exec_mode == 'single':
            
            self.style.override_matches = [0]
            self.style.scene_classes = 1
            self.style.style_classes = 1
            assert not self.style.scene_prompt, "single mode do not require prompts for scene."
            assert not self.style.style_prompt, "single mode do not require prompts for style."
            assert len(self.style.style_images) == 1, "single mode requires only 1 image."
            
        else:                    
            
            self.style.scene_classes = len(self.style.scene_prompt) + 1
            
            if self.style.exec_mode == "sematic":
                self.style.style_classes = len(self.style.style_prompt) + 1
                assert len(self.style.style_prompt) > 0, "single mode requires prompts for style."
                assert len(self.style.style_images) == 1, "sematic mode requires only 1 image."

            else:
                self.style.style_classes = len(self.style.style_images)
                assert len(self.style.style_images) > 1, "sematic mode requires at least 2 images."
                
            if not self.style.override_matches:
                self.style.override_matches = []
                for i in range(self.style.scene_classes):
                    self.style.override_matches.append(i % self.style.style_classes)
                
            assert len(self.style.override_matches) <= self.style.scene_classes, \
                   "len of override matches <= num of scene classes"
            
            assert max(self.style.override_matches) <= self.style.style_classes, \
                   "max value of override matches <= num of style classes"
            
        
    def set_debug(self, val):
        self.pipe.debug = val
    
    def _generate_output_path(self):
        if self.style.stylized_model_path is not None:
            return
        
        current_date = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
        self.style.stylized_model_path = os.path.join("output", current_date)
    

    def _save_args(self):
        
        print(f"Training on {self.model.source_path}")
        print(f"Original Gaussian model path: {self.model.model_path}")
        print(f"Stylized Gaussian model output path: {self.style.stylized_model_path}")
        
        os.makedirs(self.style.stylized_model_path, exist_ok=True)
        
        from argparse import Namespace  # for compatibility
        model_vars = vars(self.model).copy()
        with open(os.path.join(self.style.stylized_model_path, "cfg_args"), "w") as cfg_log_f:
            cfg_log_f.write(str(Namespace(**model_vars)))
            
        self.save_yaml(os.path.join(self.style.stylized_model_path, "config.yaml"))
    
    
def parse_args():
    parser = ArgumentParser(description="Stylization parameters", 
                            add_config_path_arg="config")
    
    parser.add_arguments(ModelConfig, dest="model")
    parser.add_arguments(OptimizationConfig, dest="opt")
    parser.add_arguments(PipelineConfig, dest="pipe")
    parser.add_arguments(ApplicationConfig, dest="app")
    parser.add_arguments(CheckpointConfig, dest="ckpt")
    parser.add_arguments(StyleConfig, dest="style")
    
    config = parser.parse_args()
    return ConfigManager(config)
    