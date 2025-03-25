import torch
import torchvision
from torchvision.models import VGG16_Weights
import torch.nn.functional as F
from .image import labels_downscale

class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.vgg = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).eval().to("cuda")
        self.feature_layers = [11, 13, 15]
        
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def get_features(self, original: torch.Tensor, downscale=True) -> torch.Tensor:
        image = original.unsqueeze(0)
        
        if downscale:
            image = F.interpolate(image, scale_factor=0.5, mode="bilinear")
        image = self.normalize(image)
        
        outputs = []
        final_layer = max(self.feature_layers)
        
        for idx, layer in enumerate(self.vgg.features):
            image = layer(image)
            if idx in self.feature_layers:
                outputs.append(image)
            if idx == final_layer:
                break
                
        return torch.cat(outputs, dim=1).squeeze()

    def __call__(self, images, masks, list_len, downscale=True):
        """
        @param: images: [N, C, H, W]
        @param: masks: [N, H, W]
        @param: list_len: number of classes
        
        @return: total_features_list: N list of Tensor [C, N_features]
        @return: total_features_mask: list of Tensor [new_H, new_W]
        """
        
        total_features_list = []
        total_features_mask = []
        
        for i, image in enumerate(images):
            
            feature = self.get_features(image, downscale=downscale)
            feature_mask = labels_downscale(masks[i], feature.shape[-2:])
            
            features_list = get_separated_list(feature, feature_mask, list_len)
            
            total_features_list.append(features_list)
            total_features_mask.append(feature_mask)
        
        return total_features_list, total_features_mask
    

def get_separated_list(pixels, mask, num_classes):
    separated_list = []
    for i in range(num_classes):
        separated_list.append(pixels[:, mask == i])
    return separated_list
    
    
def merge(pixels_list, num_classes):
    
    merged = [torch.tensor([], device=pixels_list[0][0].device) for _ in range(num_classes)]
    for pixels in pixels_list:
        for i in range(num_classes):
            merged[i] = torch.cat([merged[i], pixels[i]], dim=1)
    
    return merged