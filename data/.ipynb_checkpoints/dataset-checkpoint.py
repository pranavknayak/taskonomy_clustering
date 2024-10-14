import os
from typing import Union, Dict
import json

from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class TaskonomyDataset(Dataset):
    """
        Taskonomy Dataset 
            Used to load data from a specific domain (including RGB, i.e. training data), 
            for a specific building.
            Designed such that training is performed on a single building at a time.
    """
    def __init__(self, domain: str, building: str) -> None:
        super().__init__()
        self.domain_dir = f"/data/taskonomy/{domain}/taskonomy/{building}"

    def __len__(self) -> int:
        return len(os.listdir(self.domain_dir))
    
    def __getitem__(self, idx: int) -> Union[Image.Image, np.ndarray, Dict[str, str], Dict[str, int], None]:
        filename = os.listdir(self.domain_dir)[idx]
        file_extension = filename.split(".")[-1]
        if file_extension == 'png':
            return Image.open(f"{self.domain_dir}/{filename}")
        elif file_extension == 'json':
            with open(f"{self.domain_dir}/{filename}", 'r') as f:
                return json.load(f)
        elif file_extension == 'npy':
            return np.load(f"{self.domain_dir}/{filename}")
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

