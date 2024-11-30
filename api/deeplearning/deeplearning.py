from time import sleep
import cv2
from datetime import datetime
import os

from torchvision import transforms

import torch

from api.deeplearning.clients.Poster_V2Client import Poster_V2Client
from api.deeplearning.clients.DaclClient import DaclClient
from api.deeplearning.clients.DanClient import DanClient
from api.deeplearning.clients.RulClient import RulClient

from api.email_reports.send_email import send_email_full_report

import dlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

class DeepLearning_API():

    # Load the pre-trained face detector and facial landmark predictor from dlib
    detector = dlib.get_frontal_face_detector()

    # from https://github.com/ageitgey/face_recognition_models/blob/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Provide the path to your shape predictor model

    model_names = ['Default' ,'Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
   
    def eval_video(self, video_path, model_name, user_fullname, user_email, progress_func, completed_func):

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

        client = self.get_model_client(model_name)
        client.init_model()
        client.load_model()  

        faces_array = []
        cropped_faces_array = []           

        while True:
            progress_func(count_frame/(total_frames*3))
            ret, frame = rec.read()
            if not ret:
                break                                        

            faces = self.detect_faces(frame)
            faces_array.append(faces)

            count_faces = 0            
            current_cropped_faces = []
            for face in faces:
                count_faces += 1
                x, y, w, h = face.left(), face.top(), face.width(), face.height()   
                x = max(0, x)
                y = max(0, y)                  
                
                cropped_face = frame[y:face.bottom(), x:face.right()]

                # convert rgb to gray
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2GRAY)

                current_cropped_faces.append(cropped_face)

                cv2.imwrite(f'{root}/1/{count_frame}_{count_faces}.png', cropped_face)                
                # select folder after saving image because data loader needs to have the image saved

            cropped_faces_array.append(current_cropped_faces)
            
            count_frame += 1            
        
        client.select_folder(root)
        emotions, probs = client.evaluate_model(progress_func=progress_func) 

        # TODO: Call report creation here before pop happens  
        filename_report, current_datetime_report, most_common_emotion = self.full_report(emotions, current_datetime, probs, report=['pai'])
        send_email_full_report(filename_report, current_datetime_report, most_common_emotion, user_fullname, user_email)

        # iterate over faces_array, for each face in face_array, take an emotion and put it in a list, so that emotions is a list of lists
        emotions_array = []
        probs_array = []
        for faces in faces_array:
            emotions_list = []
            probs_list = []
            
            for _ in faces:
                current_emotion = emotions.pop(0)
                emotions_list.append(current_emotion)

                current_prob = probs.pop(0)
                probs_list.append(current_prob)
            
            emotions_array.append(emotions_list)
            probs_array.append(probs_list)

        # re read video
        rec = cv2.VideoCapture(video_path)
        
        for faces, cropped_faces, emotions, probs in zip(faces_array, cropped_faces_array, emotions_array, probs_array):
            progress_func(1/3 + count_frame/(total_frames*3))

            ret, frame = rec.read()
            if not ret:
                break

            result_original.write(frame)

            for face, croppped_face, emotion, prob in zip(faces, cropped_faces, emotions, probs):
                x, y, w, h = face.left(), face.top(), face.width(), face.height() 
                x = max(0, x)
                y = max(0, y)                                 
                
                if prob > 0.95:
                    color = (0, 255, 0)
                elif prob > 0.8:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{emotion} - {prob:.2%}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)                   

            result_labeled.write(frame)

            count_frame += 1            
        
        # release video writer
        result_original.release()
        result_labeled.release()

        progress_func(1.0)
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

    def get_available_models(self):
        return self.model_names
    
    def get_model_client(self, model_name):       
        if model_name == 'Fear':
            return DanClient()
        elif model_name == 'Disgust':
            return RulClient()
        else:                    
            return Poster_V2Client() 
        
    def full_report(self, emotions, current_datetime, probs, report=['pai', 'bar', 'time']):
        
        os.mkdir(f'reports/full_reports/{current_datetime}')

        emotions_df = pd.DataFrame(emotions)
        filename = f"reports/full_reports/{current_datetime}/full_report.pdf"

        most_common_emotion = emotions_df[0].value_counts().idxmax()

        with matplotlib.backends.backend_pdf.PdfPages(filename) as pdf:    
            if 'pai' in report:
                fig = plt.figure(figsize=(12,6))
                patches, texts, _ = plt.pie(emotions_df[0].value_counts(), labels=emotions_df[0].value_counts().index, autopct='%1.2f%%')
                percents = 100.*emotions_df[0].value_counts()/emotions_df[0].value_counts().sum()
                labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(emotions_df[0].value_counts().index, percents)]

                sort_legend = True
                if sort_legend:
                    patches, labels, dummy =  zip(*sorted(zip(patches, labels, emotions_df[0].value_counts()),
                                                        key=lambda x: x[2],
                                                        reverse=True))

                plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.),
                        fontsize=8)

                plt.title('Emotion Distribution')
                plt.show()
                fig.savefig(f'reports/full_reports/{current_datetime}/full_report_pie_chart.png')
                pdf.savefig(fig)

            if 'bar' in report:
                emotion_data = {'emotion': ['happy', 'sad', 'neutral', 'surprise', 'angry', 'fear', 'disgust'],
                                'amount': []}
                

                for emotion in emotion_data['emotion']:
                    if emotion in emotions_df.value_counts():
                        emotion_data['amount'].append(emotions_df.value_counts()[emotion])
                    else:
                        emotion_data['amount'].append(0)

                emotions_df2 = pd.DataFrame(emotion_data)
                emotions_df2.plot.bar(x='emotion', y='amount', rot=0)        
                fig = sns.barplot(pd.DataFrame(emotions_df2, columns=['emotion', 'count']), x='emotion', y='count')
                plt.title('Occurrences of Emotions')
                #plt.bar(emotions_df['emotion'].value_counts()[0])

                # Display numbers above the bars
                for index, row in emotions_df2.iterrows():
                    plt.text(index, row['amount'], row['amount'], color='black', ha="center")                
                    
                plt.show()     
                fig.figure.savefig(f'reports/full_reports/{current_datetime}/full_reports_emotions_occurrences.png')   
                pdf.savefig(fig.figure)

            if 'time' in report:
                emotions_df3 = emotions_df[0].map({'happy': 1, 'sad':2, 'neutral':3, 'surprise':4, 'angry':5, 'fear':6, 'disgust':7})
                y_vals = [1, 2, 3, 4, 5, 6, 7]
                y_labels = ['happy', 'sad', 'neutral', 'surprise', 'angry', 'fear', 'disgust']
                fig = plt.figure(figsize=(12,6))
                plt.plot(emotions_df3, '*')    
                plt.yticks(y_vals, y_labels)  
                plt.xlabel('Frame Number')  
                plt.ylabel('Emotion')
                plt.title('Emotion Per Frame')
                plt.show()
                fig.savefig(f'reports/quick_reports/{current_datetime}/quick_report_emotion_per_frame.png')           
                pdf.savefig(fig)        

        return filename, current_datetime, most_common_emotion                