import os
import cv2

def main():                 

    original_database_path = 'datasets/fer2013'
    new_database_path = 'datasets/fer2013_480x480'

    EMOTIONS_LIST = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    if os.path.isdir(new_database_path):
        print('Please remove beforehand the old database')
        exit(0)

    os.mkdir(new_database_path)

    original_db_test = os.path.join(original_database_path, 'test')
    original_db_train = os.path.join(original_database_path, 'train')

    new_db_test = os.path.join(new_database_path, 'test')
    new_db_train = os.path.join(new_database_path, 'train')

    os.mkdir(new_db_test)
    for emotion in EMOTIONS_LIST:
        os.mkdir(os.path.join(new_db_test, emotion))

    rescale_db(original_db_test, new_db_test)

    os.mkdir(new_db_train)
    for emotion in EMOTIONS_LIST:
        os.mkdir(os.path.join(new_db_train, emotion))

    rescale_db(original_db_train, new_db_train)


def rescale_db(original_database_path, new_database_path):
    for foldername in os.listdir(original_database_path):
        original_class_path = os.path.join(original_database_path, foldername)
        new_class_path = os.path.join(new_database_path, foldername)
        for filename in os.listdir(original_class_path):
            img_path = os.path.join(original_class_path, filename)
            img = cv2.imread(img_path)
            new_img = cv2.resize(img, [480, 480])   
            new_img_path = os.path.join(new_class_path, filename)    
            cv2.imwrite(new_img_path, new_img)    