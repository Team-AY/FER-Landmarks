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

from pygrabber.dshow_graph import FilterGraph

import matplotlib.backends.backend_pdf

from datetime import datetime

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

class Landmarks_API():
    CLF_DIR = 'models/classifiers/relative_XY_Concat_20240901160507'
    display_update_time = 0
    sample_rate = None
    current_frames = 0
    face_emotions = []
    result_original = None
    result_labeled = None
    current_datetime = None

    with open(os.path.join(CLF_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    with open(os.path.join(CLF_DIR, 'fitted_classifiers.pkl'), 'rb') as f:
        fitted_classifiers = pickle.load(f)

    with open(os.path.join(CLF_DIR, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)   

    emotions_list = []        

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

        self.current_datetime = datetime.today().strftime('%Y%m%d%H%M%S')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        if self.fps == 0:
            self.fps = 30
        if(self.display_update_time < 0.5):
            self.sample_rate = 1
        else:
            self.sample_rate = int(self.fps*self.display_update_time)

        frame_width = int(self.cap.get(3)) 
        frame_height = int(self.cap.get(4)) 

        # set resoultion to full hd
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        size = (frame_width, frame_height) 

        os.mkdir(f'videos/quick_videos/{self.current_datetime}')

        # Below VideoWriter object will create 
        # a frame of above defined The output  
        # is stored in 'filename.avi' file. 
        self.result_original = cv2.VideoWriter(f'videos/quick_videos/{self.current_datetime}/filename_original.mp4',  
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                10, size) 

        self.result_labeled = cv2.VideoWriter(f'videos/quick_videos/{self.current_datetime}/filename_labeled.mp4',  
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                10, size)   

        self.emotions_list = []      

    def close_camera(self):
        self.cap.release()
        self.current_frames = 0
        self.face_emotions = []        

    def get_frame_from_camera(self):
        ret, frame = self.cap.read()

        return ret, frame

    def get_available_cameras(self):
        devices = FilterGraph().get_input_devices()

        available_cameras = []

        for device_index, device_name in enumerate(devices):
            available_cameras.append(f"{device_index}. {device_name}")

        return available_cameras

    def classify_image(self, image):
        self.result_original.write(image)

        faces, faces_landmarks = self.detect_faces_and_landmarks(image)

        if self.current_frames % self.sample_rate == 0:
            self.face_emotions = []

        face_index = -1
        for face, landmarks in zip(faces, faces_landmarks):
            face_index += 1    

            x, y, w, h = face.left(), face.top(), face.width(), face.height()        
            landmarks = np.array(landmarks)
            landmarks = landmarks.reshape(1,-1)
            landmarks = self.scaler.transform(landmarks)            

            if self.current_frames % self.sample_rate == 0:
                emotion = self.fitted_classifiers['LDA'].__clf__.predict(landmarks)
                emotion = self.le.inverse_transform(emotion)
                self.face_emotions.append(emotion)

            else:
                if face_index < len(self.face_emotions):
                    emotion = self.face_emotions[face_index]
                else:                   
                    emotion = self.fitted_classifiers['LDA'].__clf__.predict(landmarks)
                    emotion = self.le.inverse_transform(emotion)                                        

            self.emotions_list.append(emotion)


            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        self.current_frames += 1
        self.result_labeled.write(image)

        return image        

    def quick_report(self, report = ['bar', 'time']):
        matplotlib.use('tkagg')        
        
        os.mkdir(f'reports/quick_reports/{self.current_datetime}')

        emotions_df = pd.DataFrame(self.emotions_list)
        filename = f"reports/quick_reports/{self.current_datetime}/quick_report.pdf"

        try:
            most_common_emotion = emotions_df[0].value_counts().idxmax()
        except:
            most_common_emotion = 'No Emotions Detected'

        emotions_names = ['Happiness', 'Sadness', 'Neutral', 'Surprise', 'Anger', 'Fear', 'Disgust']
        emotion_frontend_names = ['happy', 'sad', 'neutral', 'surprise', 'angry', 'fear', 'disgust']

        plt.style.use('ggplot')
        with matplotlib.backends.backend_pdf.PdfPages(filename) as pdf:            
            if 'bar' in report:
                emotion_data = {'emotion': emotions_names,
                                'amount': []}
                

                for emotion in emotion_frontend_names:
                    if emotion in emotions_df.value_counts():
                        emotion_data['amount'].append(emotions_df.value_counts()[emotion])
                    else:
                        emotion_data['amount'].append(0)

                emotions_df2 = pd.DataFrame(emotion_data)
                #emotions_df2.plot.bar(x='emotion', y='amount', rot=0)        
                #fig, ax = sns.barplot(pd.DataFrame(emotions_df2, columns=['emotion', 'count']), x='emotion', y='count')
                fig, ax = plt.subplots(figsize=(18,9))
                bars = ax.bar(emotion_data['emotion'], emotion_data['amount'], color='blue', alpha=0.7)

                max_value = max(emotion_data['amount'])
                delta = max_value/8

                image_paths = []
                for emotion_name in emotions_names:
                    image_paths.append(f'images/emojis/{emotion_name}.png')    

                # Add Images and Labels Together for X-Ticks
                for i, (x, img_path, label) in enumerate(zip(range(len(emotions_names)), image_paths, emotions_names)):
                    # Read and resize the image
                    img = Image.open(img_path)  # Image as a Pillow object
                    resized_img = img.resize((3,3))  # Resize using Pillow
                    imagebox = OffsetImage(img ,zoom=0.16)  # Adjust zoom for image size
                    ab = AnnotationBbox(imagebox, (x, -delta/2), frameon=False, box_alignment=(0.5, 1.0))  # Place near the bottom of the axis
                    ax.add_artist(ab)
                    
                    # Add the text part of the x-tick label
                    ax.text(x, -delta/4, label, ha="center", va="top", fontsize=14, fontweight='bold')  # Position text close to the image

                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.set_ylabel('Amount', fontsize=14, fontweight='bold')
                ax.set_title('Occurrences of Emotions', fontsize=24,  fontweight='bold')                
                #ax.set_xticks(x + width, emotions_names, fontsize=14, fontweight='bold')
                ax.set_xticks([])
                plt.yticks(fontsize=14, fontweight='bold')

                ax.set_ylim(-delta)
                ax.axhline(y=0, color="black", linestyle="-", linewidth=1)                
                ax.legend(loc='best', ncols=3, fontsize=14)
                
                #plt.bar(emotions_df['emotion'].value_counts()[0])

                # Display numbers above the bars
                for index, row in emotions_df2.iterrows():
                    plt.text(index, row['amount'], row['amount'], color='black', ha="center",
                             fontsize=14, fontweight='bold')                
                    
                plt.show()     
                fig.figure.savefig(f'reports/quick_reports/{self.current_datetime}/quick_report_emotions_occurrences.png')   
                pdf.savefig(fig.figure)

            if 'time' in report:
                try:
                    emotions_df3 = emotions_df[0].map({'happy': 1, 'sad':2, 'neutral':3, 'surprise':4, 'angry':5, 'fear':6, 'disgust':7})
                except:
                    emotions_df3 = pd.Series([0, 0, 0, 0, 0, 0, 0])
                y_vals = [1, 2, 3, 4, 5, 6, 7]
                y_labels = ['happy', 'sad', 'neutral', 'surprise', 'angry', 'fear', 'disgust']
                fig = plt.figure(figsize=(18,9))
                plt.plot(emotions_df3, '*')    
                plt.yticks(y_vals, y_labels, fontsize=14, fontweight='bold')  
                plt.xticks(fontsize=14, fontweight='bold')
                plt.xlabel('Frame Number', fontsize=14, fontweight='bold')  
                plt.ylabel('Emotion', fontsize=14, fontweight='bold')
                plt.title('Emotion Per Frame', fontsize=24, fontweight='bold')
                plt.show()
                fig.savefig(f'reports/quick_reports/{self.current_datetime}/quick_report_emotion_per_frame.png')           
                pdf.savefig(fig)

        return filename, self.current_datetime, most_common_emotion