import cv2
import dlib
import math
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import LabelEncoder

class Landmarks_API():
    CLF_DIR = 'models/classifiers/relative_XY_Concat_20240901160507'

    with open(os.path.join(CLF_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    # Load the pre-trained face detector and facial landmark predictor from dlib
    detector = dlib.get_frontal_face_detector()

    # from https://github.com/ageitgey/face_recognition_models/blob/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Provide the path to your shape predictor model

    # Function to detect faces and facial landmarks
    def detect_faces_and_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        faces_landmarks = []
        for face in faces:

            landmarks = self.predictor(gray,face)
            BB_x, BB_y, BB_w, BB_h = face.left(), face.top(), face.width(), face.height()
            landmarks_points_x = [(landmarks.part(i).x-BB_x)/BB_w for i in range(68)]
            landmarks_points_y = [(landmarks.part(i).y-BB_y)/BB_h for i in range(68)]
            landmarks_points = landmarks_points_x + landmarks_points_y

            faces_landmarks.append(landmarks_points)

            

        return faces, faces_landmarks
    
    def open_camera(self, camera_num):
        self.cap = cv2.VideoCapture(camera_num)
        self.cap.getBackendName()

        frame_width = int(self.cap.get(3)) 
        frame_height = int(self.cap.get(4)) 

        # set resoultion to full hd
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        size = (frame_width, frame_height) 

        # Below VideoWriter object will create 
        # a frame of above defined The output  
        # is stored in 'filename.avi' file. 
        self.result_original = cv2.VideoWriter('videos/filename_original.avi',  
                                cv2.VideoWriter_fourcc(*'MJPG'), 
                                10, size) 

        self.result_labeled = cv2.VideoWriter('videos/filename_labeled.avi',  
                                cv2.VideoWriter_fourcc(*'MJPG'), 
                                10, size)         

    def close_camera(self):
        self.cap.release()

    def get_frame_from_camera(self):
        ret, frame = self.cap.read()

        return ret, frame

    def list_cameras(self):
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            try:
                name = cap.getBackendName()                
                arr.append((index, name))
            except:
                break
            cap.release()
            index += 1

        return arr

    with open(os.path.join(CLF_DIR, 'fitted_classifiers.pkl'), 'rb') as f:
        fitted_classifiers = pickle.load(f)

    with open(os.path.join(CLF_DIR, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)   

    emotions_list = []

    def classify_image(self, image):
        #result_original.write(frame)

        faces, faces_landmarks = self.detect_faces_and_landmarks(image)

        for face, landmarks in zip(faces, faces_landmarks):
            x, y, w, h = face.left(), face.top(), face.width(), face.height()        
            landmarks = np.array(landmarks)
            landmarks = landmarks.reshape(1,-1)
            landmarks = self.scaler.transform(landmarks)
            emotion = self.fitted_classifiers['QDA'].__clf__.predict(landmarks)
            emotion = self.le.inverse_transform(emotion)
            self.emotions_list.append(emotion)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)



        return image

        #result_labeled.write(frame)

