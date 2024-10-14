from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchinfo import summary
from data.dataset import TaskonomyDataset
import torch
import torch.nn as nn

from models import backbone
from models import decoder
import building_selection, task_selection

import os
import argparse

taskonomy_base_path = "/data/taskonomy"


def get_optimizir(optimizer_name, learning_rate, model):
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on Taskonomy data, for a specific building and domain")
    parser.add_argument("--domain", type=str, default="rgb", help="The domain to train on, defaults to autoencoder targets (i.e. the training data).")
    parser.add_argument("--epochs", type=int, required=True, help="The number of epochs to train for")
    parser.add_argument("--batch_size", type=int, required=True, help="The batch size to use for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="The learning rate to use for training")
    parser.add_argument("--optimizer", type=str, required=True, help="The optimizer to use for training")
    args = parser.parse_args()
    backbone = ResnetBackbone(pretrained=True).to("cuda")
    
    for building in building_selection.paths:
        train_dataset = TaskonomyDataset("rgb", building)
        test_dataset = TaskonomyDataset(args.domain, building)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
