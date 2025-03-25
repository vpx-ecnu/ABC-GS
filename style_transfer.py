
import sys
sys.path.append("./gs")

import torch
from abcgs.configs import parse_args
from gs.utils.general_utils import safe_state
from gs.gaussian_renderer import network_gui
from abcgs.trainer import StyleTrainer
import random

def main():
    
    config = parse_args()
    
    random.seed(config.style.random_seed)
    torch.manual_seed(config.style.random_seed)
    torch.cuda.manual_seed_all(config.style.random_seed)
    
    safe_state(config.app.quiet)
    network_gui.init(config.app.ip, config.app.port)
    torch.autograd.set_detect_anomaly(config.app.detect_anomaly)
    
    trainer = StyleTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()

