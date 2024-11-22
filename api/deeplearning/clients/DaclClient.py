from api.deeplearning.clients.BaseClient import BaseClient

from torchvision import transforms

import torch
from ..models.dacl import resnet18
import os


class DaclClient(BaseClient):
    checkpoint_path = os.path.join('.', r'api\deeplearning\checkpoints\dacl\fer2013_batch-128_fernorm.pth')
    model = None
    def select_folder(self, root=None):
        if root is None or not os.path.exists(root):
            raise Exception("Invalid folder path")

        rafnormalize = transforms.Normalize(mean=[0.5752, 0.4495, 0.4012],
                                            std=[0.2086, 0.1911, 0.1827])  

        self.data_loader = self.create_data_loader(root = root, bs = 64, workers=2, normalize=rafnormalize)

    def init_model(self):
        self.model = resnet18(pretrained='msceleb')
        self.model.fc = torch.nn.Linear(512, 7)
        self.model = torch.nn.DataParallel(self.model)

    def evaluate_model(self, progress_func=None):
        all_preds = []
        all_labels = []

        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            num_frames = len(self.data_loader.dataset)
            for images, labels in self.data_loader:  # Loop through batches
                images = images.to(self.device) # Move images to the device
                labels = labels.to(self.device) # Move labels to the device
                _, predictions, *_ = self.model(images)
                _, predicted_labels = torch.max(predictions, 1)

                all_preds.extend(predicted_labels.cpu().numpy())  # Store predictions
                all_labels.extend(labels.cpu().numpy())  # Store actual labels

                if progress_func is not None:
                    progress_func(1/3 + (len(all_preds) / num_frames)/3)


        return all_preds    

if __name__ == "__main__":      

    client = DaclClient()
    client.init_model()
    client.load_model()    
    all_preds = client.evaluate_model()
    print(all_preds)