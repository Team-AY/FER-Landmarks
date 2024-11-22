from time import sleep
import cv2
from datetime import datetime
import os

from torchvision import transforms

import torch

from api.deeplearning.clients.DaclClient import DaclClient

import dlib

class DeepLearning_API():

    # Load the pre-trained face detector and facial landmark predictor from dlib
    detector = dlib.get_frontal_face_detector()

    # from https://github.com/ageitgey/face_recognition_models/blob/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Provide the path to your shape predictor model


    def eval_video(self, video_path, progress_func, completed_func):

        completed_func(False)
        
        rec = cv2.VideoCapture(video_path)
        current_datetime = datetime.today().strftime('%Y%m%d%H%M%S')
        os.mkdir(f'temp/{current_datetime}')
        # 1 is because data loader needs any class folder
        os.mkdir(f'temp/{current_datetime}/1')

        count_frame = 1
        total_frames = rec.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = rec.get(cv2.CAP_PROP_FPS)
        root = f'temp/{current_datetime}'

        frame_width = int(rec.get(3)) 
        frame_height = int(rec.get(4)) 

        size = (frame_width, frame_height)

        filename = os.path.splitext(os.path.basename(video_path))[0]

        os.mkdir(f'videos/{current_datetime}')

        result_original = cv2.VideoWriter(f'videos/{current_datetime}/{filename}_original.avi',  
                                cv2.VideoWriter_fourcc(*'MJPG'), 
                                fps, size)  

        result_labeled = cv2.VideoWriter(f'videos/{current_datetime}/{filename}_labeled.avi',  
                                cv2.VideoWriter_fourcc(*'MJPG'), 
                                fps, size)  

        client = DaclClient()        
        client.init_model()
        client.load_model()  

        frames_array = []
        faces_array = []
        cropped_faces_array = []           

        while True:
            progress_func(count_frame/(total_frames*3))
            ret, frame = rec.read()
            if not ret:
                break                            

            frames_array.append(frame)

            faces = self.detect_faces(frame)
            faces_array.append(faces)

            count_faces = 0            
            current_cropped_faces = []
            for face in faces:
                count_faces += 1
                x, y, w, h = face.left(), face.top(), face.width(), face.height()     
                
                cropped_face = frame[face.top():face.bottom(), face.left():face.right()]
                current_cropped_faces.append(cropped_face)

                cv2.imwrite(f'{root}/1/{count_frame}_{count_faces}.png', cropped_face)                
                # select folder after saving image because data loader needs to have the image saved

            cropped_faces_array.append(current_cropped_faces)
            
            count_frame += 1            
        
        client.select_folder(root)
        emotions = client.evaluate_model(progress_func=progress_func)   

        for frame, faces, cropped_faces, emotion in zip(frames_array, faces_array, cropped_faces_array, emotions):
            progress_func(1/3 + count_frame/(total_frames*3))

            result_original.write(frame)

            for face, croppped_face in zip(faces, cropped_faces):
                x, y, w, h = face.left(), face.top(), face.width(), face.height() 

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)                   

            result_labeled.write(frame)

            count_frame += 1            

        # release video writer
        result_original.release()
        result_labeled.release()

        completed_func(True)
        return True

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)        

        return faces
            
    def fake_eval_frame(self, video_path, progress_func, completed_func):
        print("Fake Eval Frame")        
        percent = 0
        for i in range(10):
            sleep(0.1)
            percent = percent + 0.1
            progress_func(percent)
        
        completed_func(True)
        return True
        