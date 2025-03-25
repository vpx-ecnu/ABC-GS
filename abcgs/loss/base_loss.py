import torch
import torchvision
from torchvision.models import VGG16_Weights
from abc import abstractmethod, ABC

class StylizeLoss(torch.nn.Module, ABC):
    def __init__(self, config):
        super().__init__() 
        
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)
        
        self.vgg = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).eval().to("cuda")
        self.config = config
        
    @abstractmethod
    def __call__(self, render_feats_list, style_feats_list): ...