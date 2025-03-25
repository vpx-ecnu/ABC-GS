from lang_sam import LangSAM
from torchvision.transforms import ToPILImage
import torch

class Segmenter:
    def __init__(self):
        self.model = LangSAM()
    
    def __call__(self, image, label):
        pil_image = ToPILImage(mode="RGB")(image)
        masks, _, _, _ = self.model.predict(pil_image, label)
        masks, _ = torch.max(masks, dim=0)
        return masks