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

CLF_DIR = 'models/classifiers/relative_XY_Concat'

with open(os.path.join(CLF_DIR, 'scaler.pkl'), 'rb') as f:
   scaler = pickle.load(f)

# Load the pre-trained face detector and facial landmark predictor from dlib
detector = dlib.get_frontal_face_detector()

# from https://github.com/ageitgey/face_recognition_models/blob/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Provide the path to your shape predictor model

# Function to detect faces and facial landmarks
def detect_faces_and_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    faces_landmarks = []
    for face in faces:
        #x, y, w, h = face.left(), face.top(), face.width(), face.height()

        #end_x = x+w
        #end_y = y+h
        #if x<0:
        #    x=0
        #if y<0:
        #    y=0
        #if end_x>480:
        #    end_x=480
        #if end_y>480:
        #    end_y=480            
        #face_crop = gray[x:end_x,y:end_y]        
        #face_crop_resize = cv2.resize(face_crop, (48, 48))

        #x1, y1, x2, y2 = 0, 0 , 48 ,48
        #face_roi = dlib.rectangle(x1, y1, x2, y2)

        landmarks = predictor(gray,face)
        BB_x, BB_y, BB_w, BB_h = face.left(), face.top(), face.width(), face.height()
        #landmarks_points = [landmarks.part(i).x+landmarks.part(i).y for i in range(68)]
        landmarks_points_x = [(landmarks.part(i).x-BB_x)/BB_w for i in range(68)]
        landmarks_points_y = [(landmarks.part(i).y-BB_y)/BB_h for i in range(68)]
        landmarks_points = landmarks_points_x + landmarks_points_y

        faces_landmarks.append(landmarks_points)

        #landmarks = predictor(face_crop_resize, face_roi)
        #landmarks_points = [math.dist((landmarks.part(i).x, landmarks.part(i).y),(0,0) )for i in range(68)]
        #faces_landmarks.append(landmarks_points)
        

    return faces, faces_landmarks

# Example usage:
cap = cv2.VideoCapture(0)

with open(os.path.join(CLF_DIR, 'fitted_classifiers.pkl'), 'rb') as f:
   fitted_classifiers = pickle.load(f)

emotions_list = []

frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 

# Below VideoWriter object will create 
# a frame of above defined The output  
# is stored in 'filename.avi' file. 
result_original = cv2.VideoWriter('videos/filename_original.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size) 

result_labeled = cv2.VideoWriter('videos/filename_labeled.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size) 

while True:
    ret, frame = cap.read()

    if not ret:
        break

    result_original.write(frame)

    faces, faces_landmarks = detect_faces_and_landmarks(frame)

    for face, landmarks in zip(faces, faces_landmarks):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()        
        landmarks = np.array(landmarks)
        landmarks = landmarks.reshape(1,-1)
        landmarks = scaler.transform(landmarks)
        emotion = fitted_classifiers['LDA'].__clf__.predict(landmarks)
        emotions_list.append(emotion)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)



    cv2.imshow('Face Detection with Landmarks', frame)

    result_labeled.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

def quick_report(report = ['bar', 'time']):

    emotions_df = pd.DataFrame(emotions_list)

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
        plt.title('Occurnces of Emotions')
        #plt.bar(emotions_df['emotion'].value_counts()[0])
        plt.show()     
        fig.figure.savefig('reports/quick_reports/report_emotions_occurnces.png')   

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
        fig.savefig('reports/quick_reports/report_emotion_per_frame.png')   

quick_report(['bar', 'time'])
print("done")