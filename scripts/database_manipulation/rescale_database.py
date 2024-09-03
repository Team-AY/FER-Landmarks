import os
import cv2

original_database_path = '../../datasets/fer2013/test'
new_database_path = '../../datasets/fer2013_480x480/test'

for foldername in os.listdir(original_database_path):
    original_class_path = os.path.join(original_database_path, foldername)
    new_class_path = os.path.join(new_database_path, foldername)
    for filename in os.listdir(original_class_path):
        img_path = os.path.join(original_class_path, filename)
        img = cv2.imread(img_path)
        new_img = cv2.resize(img, [480, 480])   
        new_img_path = os.path.join(new_class_path, filename)    
        cv2.imwrite(new_img_path, new_img)

original_database_path = '../../datasets/fer2013/train'
new_database_path = '../../datasets/fer2013_480x480/train'

for foldername in os.listdir(original_database_path):
    original_class_path = os.path.join(original_database_path, foldername)
    new_class_path = os.path.join(new_database_path, foldername)
    for filename in os.listdir(original_class_path):
        img_path = os.path.join(original_class_path, filename)
        img = cv2.imread(img_path)
        new_img = cv2.resize(img, [480, 480])   
        new_img_path = os.path.join(new_class_path, filename)    
        cv2.imwrite(new_img_path, new_img)        