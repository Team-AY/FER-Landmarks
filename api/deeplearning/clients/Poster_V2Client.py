from api.deeplearning.clients.BaseClient import BaseClient

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
from ..models.poster_v2 import PosterV2_7cls

import pandas as pd


class Poster_V2Client(BaseClient):
    checkpoint_path = os.path.join('.', r'api\deeplearning\checkpoints\poster_v2\POSTER-batch-128.pth')    
    model = None
    mapper = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}
    def select_folder(self, root=None):
        if root is None or not os.path.exists(root):
            raise Exception("Invalid folder path")

        rafnormalize = transforms.Normalize(mean=[0.5752, 0.4495, 0.4012],
                                            std=[0.2086, 0.1911, 0.1827])  

        self.data_loader = self.create_data_loader(root = root, bs = 64, workers=2, normalize=rafnormalize)

    def init_model(self):
        self.model = PosterV2_7cls.pyramid_trans_expr2(img_size=224, num_classes=7)
        self.model = torch.nn.DataParallel(self.model).to(self.device)

    def evaluate_model(self, progress_func=None):
        all_preds = []
        all_labels = []      

        self.model.to(self.device)  
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            num_frames = len(self.data_loader.dataset)
            for images, labels in self.data_loader:  # Loop through batches
                images = images.to(self.device) # Move images to the device
                labels = labels.to(self.device) # Move labels to the device
                predictions = self.model(images)
                _, predicted_labels = torch.max(predictions, 1)

                all_preds.extend(predicted_labels.cpu().numpy())  # Store predictions
                all_labels.extend(labels.cpu().numpy())  # Store actual labels

                if progress_func is not None:
                    progress_func(1/3 + (len(all_preds) / num_frames)/3)


        #return all_preds    
        return list(pd.Series(all_preds).map(self.mapper).values)
    
    def load_model(self):
        """""
        Args: checkpoint_path (str): Path to the checkpoint file.
        Set model to evaluation mode
        Returns: torch.nn.Module: Loaded model.
        """""
        checkpoint = torch.load(self.checkpoint_path,map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])
        return self.model   


if __name__ == "__main__":      

    client = Poster_V2Client()
    client.init_model()
    client.load_model()    
    all_preds = client.evaluate_model()
    print(all_preds)