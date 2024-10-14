from PIL import Image
import numpy as np
from models.backbone import ResnetBackbone
if __name__ == "__main__":
    image_path = "/data/taskonomy/depth_euclidean/taskonomy/allensville/point_0_view_0_domain_depth_euclidean.png"
    resnet = ResnetBackbone(pretrained=False)
    img = Image.open(image_path).convert('RGB')
    print(type(resnet(img)))

