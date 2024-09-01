import cv2
import dlib
import os
import csv
import time
import math
import numpy as np

# Load the pre-trained face detector and facial landmark predictor from dlib
detector = dlib.get_frontal_face_detector()

# from https://github.com/ageitgey/face_recognition_models/blob/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Provide the path to your shape predictor model
#predictor = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")  # Provide the path to your shape predictor model



def main(data_type):

    # Open the CSV file for writing features in append mode
    csv_features_file = open(f"datasets/landmarks/landmarks_480x480_X_Plus_Y_features_{data_type}.csv", "w", newline="")
    csv_features_writer = csv.writer(csv_features_file)

    # Open the CSV file for writing labels in append mode
    csv_labels_file = open(f"datasets/landmarks/landmarks_480x480_X_Plus_Y_labels_{data_type}.csv", "w", newline="")
    csv_labels_writer = csv.writer(csv_labels_file)

    data_path = f'datasets/fer2013_480x480/{data_type}'


    # Function to detect faces and facial landmarks
    # analyzes 480x480 pictures and extracts landmarks as a tupple of (x,y)
    def detect_faces_and_landmarks(image, label):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if(len(faces) != 1):
            return   
             
        face = faces[0]

        landmarks = predictor(gray,face)
        #landmarks_points = [landmarks.part(i).x+landmarks.part(i).y for i in range(68)]
        #landmarks_points_x = [landmarks.part(i).x for i in range(68)]
        #landmarks_points_y = [landmarks.part(i).y for i in range(68)]
        #landmarks_points = landmarks_points_x + landmarks_points_y
        landmarks_points = [landmarks.part(i).x+landmarks.part(i).y for i in range(68)]
     
        csv_features_writer.writerow(landmarks_points)
        csv_labels_writer.writerow([label])

        return 

    # Loop through all images in the directory
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        for filename in os.listdir(label_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for image extensions
                image_path = os.path.join(label_path, filename)
                image = cv2.imread(image_path)
                detect_faces_and_landmarks(image, label)



    csv_features_file.close()
    csv_labels_file.close()


if __name__ == '__main__':
    start = time.time()
    print(start)
    main("train")
    end = time.time()
    print(end)
    elaps =end - start  
    print(elaps)
