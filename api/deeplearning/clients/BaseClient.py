from abc import ABC, abstractmethod

import torch
from torchvision import transforms
import torch.utils.data as data

import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
from PIL import Image
import os
#from models.rul.src import utils
#from models.rul.src import dataset

from types import SimpleNamespace

class BaseClient(ABC):
    checkpoint_path = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def init_model(self, checkpoint=None):
        pass

    def data_loader(self, root, bs, workers, normalize):

        val_loader = torch.utils.data.DataLoader(
                    dataset=datasets.ImageFolder(
                        root=root,
                        transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ])
                    ),
                    batch_size=bs, shuffle=False,
                    num_workers=workers, pin_memory=True
                )

        return val_loader

    def load_model(self):
        """""
        Args: checkpoint_path (str): Path to the checkpoint file.
        Set model to evaluation mode
        Returns: torch.nn.Module: Loaded model.
        """""
        checkpoint = torch.load(self.checkpoint_path,map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        return self.model