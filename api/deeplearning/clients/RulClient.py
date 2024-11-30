from api.deeplearning.clients.BaseClient import BaseClient

from torchvision import transforms

import torch
from ..models.rul import rul
import os

import pandas as pd


class RulClient(BaseClient):
    checkpoint_path = os.path.join('.', r'api\deeplearning\checkpoints\rul\RUL-batch256.pth')
    model = None
    mapper = {0: 'Surprise', 1: 'Fear', 2: 'Disgust', 3: 'Happiness', 4: 'Sadness', 5: 'Anger', 6: 'Neutral'}
    def select_folder(self, root=None):
        if root is None or not os.path.exists(root):
            raise Exception("Invalid folder path")

        rafnormalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])  

        self.data_loader = self.create_data_loader(root = root, bs = 64, workers=2, normalize=rafnormalize)

    def init_model(self):
        self.model = rul.res18feature(args=None, pretrained=False, num_classes=7, drop_rate=0.4, out_dim=64)
        self.fc = torch.nn.Linear(64, 7)                

    def evaluate_model(self, progress_func=None):
        all_preds = []
        all_labels = []

        self.fc.to(self.device)
        self.model.to(self.device)

        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            num_frames = len(self.data_loader.dataset)
            for images, labels in self.data_loader:  # Loop through batches
                images = images.to(self.device) # Move images to the device
                labels = labels.to(self.device) # Move labels to the device
                predictions = self.model(images, labels, phase='test')
                outputs = self.fc(predictions)
                _, predicted_labels = torch.max(outputs, 1)                

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
        self.model.load_state_dict(checkpoint['model_state_dict'])        
        self.fc.load_state_dict(checkpoint['fc_state_dict'])        
        
        return self.model    

if __name__ == "__main__":      

    client = RulClient()
    client.init_model()
    client.load_model()    
    all_preds = client.evaluate_model()
    print(all_preds)