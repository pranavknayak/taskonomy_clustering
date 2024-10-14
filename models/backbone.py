import torch
import torch.nn as nn
from transformers import ResNetModel, AutoImageProcessor
from PIL import Image

class ResnetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResnetBackbone, self).__init__()
        self.model = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    def forward(self, x: Image.Image): # Preprocessor moves images to cuda, dw
        inputs = self.processor(x, return_tensors="pt").to("cuda")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state
