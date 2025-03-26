import sys
sys.path.append("./gs")
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)
sys.path.append(workspace_dir)

from abcgs.utils.image import render_viewpoint
from abcgs.configs import parse_args
from abcgs.trainer import StyleTrainer

    
if __name__ == '__main__':
    
    config = parse_args()
    config.model.model_path = config.style.stylized_model_path
    
    trainer = StyleTrainer(config)
    render_viewpoint(trainer, config.model.model_path)

