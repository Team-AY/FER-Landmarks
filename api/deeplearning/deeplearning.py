from time import sleep
import cv2
from datetime import datetime
import os

from torchvision import transforms

import torch

from api.deeplearning.clients.DaclClient import DaclClient

class DeepLearning_API():
    def eval_video(self, video_path, progress_func, completed_func):
        
        rec = cv2.VideoCapture(video_path)
        current_datetime = datetime.today().strftime('%Y%m%d%H%M%S')
        os.mkdir(f'temp/{current_datetime}')
        # 1 is because data loader needs any class folder
        os.mkdir(f'temp/{current_datetime}/1')

        count_frame = 1
        total_frames = rec.get(cv2.CAP_PROP_FRAME_COUNT)
        root = f'temp/{current_datetime}'
        while True:
            progress_func(count_frame/(total_frames*2))
            ret, frame = rec.read()
            if not ret:
                break
            
            cv2.imwrite(f'{root}/1/{count_frame}.png', frame)
            count_frame += 1            

        # TODO: Call the model to predict emotions

        client = DaclClient(root=root)
        client.init_model()
        client.load_model()    
        all_preds = client.evaluate_model(progress_func=progress_func)
        print(all_preds)        
        return True
        #return self.fake_eval_frame(video_path, progress_func, completed_func)

    def fake_eval_frame(self, video_path, progress_func, completed_func):
        print("Fake Eval Frame")        
        percent = 0
        for i in range(10):
            sleep(0.1)
            percent = percent + 0.1
            progress_func(percent)
        
        completed_func(True)
        return True
        