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

import numpy as np

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

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

        os.mkdir(f'videos/full_videos/{current_datetime}')

        result_original = cv2.VideoWriter(f'videos/full_videos/{current_datetime}/{filename}_original.avi',  
                                cv2.VideoWriter_fourcc(*'MJPG'), 
                                fps, size)  

        result_labeled = cv2.VideoWriter(f'videos/full_videos/{current_datetime}/{filename}_labeled.avi',  
                                cv2.VideoWriter_fourcc(*'MJPG'), 
                                fps, size)  

        client = self.get_model_client(model_name)
        client.init_model()
        client.load_model()  

        faces_array = []          

        while True:
            progress_func(count_frame/(total_frames*3))
            ret, frame = rec.read()
            if not ret:
                break                                        

            faces = self.detect_faces(frame)
            faces_array.append(faces)

            count_faces = 0                        
            for face in faces:
                count_faces += 1
                x, y, w, h = face.left(), face.top(), face.width(), face.height()   
                x = max(0, x)
                y = max(0, y)                  
                
                cropped_face = frame[y:face.bottom(), x:face.right()]

                # convert rgb to gray
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2GRAY)                

                cv2.imwrite(f'{root}/1/{count_frame}_{count_faces}.png', cropped_face)                
                # select folder after saving image because data loader needs to have the image saved            
            
            count_frame += 1            
        
        client.select_folder(root)
        emotions, probs = client.evaluate_model(progress_func=progress_func) 

        # TODO: Call report creation here before pop happens  
        filename_report, current_datetime_report, most_common_emotion = self.full_report(emotions, current_datetime, probs, report=['pai', 'bar', 'time'])
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
        
        for faces, emotions, probs in zip(faces_array, emotions_array, probs_array):
            progress_func(1/3 + count_frame/(total_frames*3))

            ret, frame = rec.read()
            if not ret:
                break

            result_original.write(frame)

            for face, emotion, prob in zip(faces, emotions, probs):
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
        matplotlib.use('Agg')
        os.mkdir(f'reports/full_reports/{current_datetime}')

        emotions_df = pd.DataFrame(emotions)
        probs_df = pd.DataFrame(probs)
        probs_df = probs_df.applymap(mapper_func)
        filename = f"reports/full_reports/{current_datetime}/full_report.pdf"

        most_common_emotion = emotions_df[0].value_counts().idxmax()

        #plt.show()

        plt.style.use('ggplot')
        with matplotlib.backends.backend_pdf.PdfPages(filename) as pdf:    
            if 'pai' in report:
                #emotions
                plt.style.use('default')

                data = emotions_df[0].value_counts().to_dict()

                base_d = sum(list(data.values()))
                final_data = {k:m/base_d*100 for k,m in data.items()}

                fig, ax = plt.subplots(figsize=(18,9), subplot_kw=dict(aspect="equal"))
                recipe = list(final_data.keys())
                data = list(final_data.values())
                perc = [str(round(e / s * 100., 2)) + '%' for s in (sum(data),) for e in data]
                wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40, textprops={'fontweight': 'bold', 'fontsize': 14})
                bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
                kw = dict(arrowprops=dict(arrowstyle="-"),
                        #bbox=bbox_props,
                        zorder=0, va="center",
                        fontsize=14, fontweight='bold')

                for i, p in enumerate(wedges):
                    ang = (p.theta2 - p.theta1)/2. + p.theta1
                    y = np.sin(np.deg2rad(ang))
                    x = np.cos(np.deg2rad(ang))
                    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
                    kw["arrowprops"].update({"connectionstyle": connectionstyle})
                    ax.annotate(recipe[i] + ' ' + perc[i], xy=(x, y), xytext=(1.4*np.sign(x), 1.4*y),
                                horizontalalignment=horizontalalignment, **kw)


                #fig = plt.figure(figsize=(18,9))                
                patches = wedges#, texts, _ = plt.pie(emotions_df[0].value_counts(), labels=emotions_df[0].value_counts().index, autopct='%1.2f%%', textprops={'fontweight': 'bold', 'fontsize': 14})
                percents = 100.*emotions_df[0].value_counts()/emotions_df[0].value_counts().sum()
                labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(emotions_df[0].value_counts().index, percents)]

                sort_legend = True
                if sort_legend:
                    patches, labels, dummy =  zip(*sorted(zip(patches, labels, emotions_df[0].value_counts()),
                                                        key=lambda x: x[2],
                                                        reverse=True))
                    
                plt.gcf().legend(patches, labels, loc='center left', fontsize=14)                    
                
                # change style
                circle = plt.Circle(xy=(0,0), radius=0.80, facecolor='white')
                plt.gca().add_artist(circle)                  

                plt.title('Emotion Distribution', fontsize=24,  fontweight='bold')
                
                #plt.show()                

                fig.savefig(f'reports/full_reports/{current_datetime}/full_report_emotion_pie_chart.png')
                pdf.savefig(fig)

                plt.style.use('ggplot')

                #probs
                colors = []
                for index in probs_df[0].value_counts().index:
                    if index == 'High':
                        colors.append('g')
                    elif index == 'Medium':
                        colors.append('yellow')
                    else:
                        colors.append('r')

                fig = plt.figure(figsize=(18,9)) 
                patches, texts, _ = plt.pie(probs_df[0].value_counts(), labels=probs_df[0].value_counts().index, 
                                            colors=colors, autopct='%1.2f%%', 
                                            textprops={'fontweight': 'bold', 'fontsize': 14})
                percents = 100.*probs_df[0].value_counts()/probs_df[0].value_counts().sum()
                labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(probs_df[0].value_counts().index, percents)]

                sort_legend = True
                if sort_legend:
                    patches, labels, dummy =  zip(*sorted(zip(patches, labels, probs_df[0].value_counts()),
                                                        key=lambda x: x[2],
                                                        reverse=True))

                plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.),
                        fontsize=14)              

                plt.title('Probability Distribution', fontsize=24,  fontweight='bold')
                #plt.show()
                fig.savefig(f'reports/full_reports/{current_datetime}/full_report_probability_pie_chart.png')
                pdf.savefig(fig)                

            if 'bar' in report:
            
                emotions_names = ('Happiness', 'Sadness', 'Neutral', 'Surprise', 'Anger', 'Fear', 'Disgust')
                colors = ['g', 'yellow', 'r']
                probability_amount = {
                    'High': (),
                    'Medium': (),
                    'Low': (),
                }
                for emotion_name in emotions_names:
                    for probability in probability_amount.keys():
                        probability_amount[probability] += (probs_df[emotions_df[0] == emotion_name] == probability).sum()[0],

                x = np.arange(len(emotions_names))  # the label locations
                width = 0.25  # the width of the bars
                multiplier = 0

                fig, ax = plt.subplots(figsize=(18,9))  

                for attribute, measurement in probability_amount.items():
                    offset = width * multiplier
                    rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[multiplier])
                    ax.bar_label(rects, padding=3, fontsize=14, fontweight='bold')                    
                    multiplier += 1

                max_value = np.array(list(probability_amount.values())).max()
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
                    ab = AnnotationBbox(imagebox, (x + width, -delta/2), frameon=False, box_alignment=(0.5, 1.0))  # Place near the bottom of the axis
                    ax.add_artist(ab)
                    
                    # Add the text part of the x-tick label
                    ax.text(x + width, -0.5, label, ha="center", va="top", fontsize=14, fontweight='bold')  # Position text close to the image

                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.set_ylabel('Amount', fontsize=14, fontweight='bold')
                ax.set_title('Probability Distribution by Emotions', fontsize=24,  fontweight='bold')
                #ax.set_xticks(x + width, emotions_names, fontsize=14, fontweight='bold')
                ax.set_xticks([])
                plt.yticks(fontsize=14, fontweight='bold')

                ax.set_ylim(-delta)
                ax.axhline(y=0, color="black", linestyle="-", linewidth=1)                
                ax.legend(loc='best', ncols=3, fontsize=14)                

                #plt.show()
                fig.figure.savefig(f'reports/full_reports/{current_datetime}/full_reports_probability_distribution.png')   
                pdf.savefig(fig.figure)                

            if 'time' in report:
                emotions_df3 = emotions_df[0].map({'Happiness': 1, 'Sadness':2, 'Neutral':3, 'Surprise':4, 'Angrer':5, 'Fear':6, 'Disgust':7})
                y_vals = [1, 2, 3, 4, 5, 6, 7]
                y_labels = ['Happiness', 'Sadness', 'Neutral', 'Surprise', 'Anger', 'Fear', 'Disgust']
                fig = plt.figure(figsize=(18,9))
                plt.plot(emotions_df3, '*')    
                plt.yticks(y_vals, y_labels, fontsize=14, fontweight='bold')  
                plt.xlabel('Frame Number', fontsize=14, fontweight='bold')  
                plt.ylabel('Emotion', fontsize=14, fontweight='bold')
                plt.xticks(fontsize=14, fontweight='bold')
                plt.title('Emotion Per Frame', fontsize=24,  fontweight='bold')
                #plt.show()
                fig.savefig(f'reports/full_reports/{current_datetime}/full_report_emotion_per_frame.png')           
                pdf.savefig(fig)        

        return filename, current_datetime, most_common_emotion                
    
def mapper_func(percent):

    if percent<0.80:
        return "Low"

    elif percent<0.95:
        return "Medium"

    else:
        return "High"    