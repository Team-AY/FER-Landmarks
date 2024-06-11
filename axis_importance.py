import pandas as pd 
import sklearn
import math

import sklearn.utils
from model import Model
import matplotlib.pyplot as plt
import pickle
import numpy as np
import landmarks_annot as lm
import ast


#Prepare Data
converters = {col: ast.literal_eval for col in range(68)}
#load the train data as dataframe
Train_data = pd.read_csv("datasets/landmarks/landmarks_features_points_train.csv" , header=None, converters=converters)
#Train_data = Train_data.values #get values

#load the train labels as dataframe
Train_labels = pd.read_csv("datasets/landmarks/landmarks_labels_points_train.csv", header=None)
#Train_labels = Train_labels.values.ravel() #set the right shape


emotions = np.unique(Train_labels)
emotions_dict = {}
for emotion in emotions:
    
    condition_df = Train_labels==emotion
    data = Train_data[condition_df[0]]
    features_avg = []
    for i in range(68):
        test = np.array(data[i].values)
        a = [*zip(*test)]
        b = [list(i) for i in zip(*test)]
        test1 = np.array(b)
        features_avg.append(test1.mean(1))

    emotions_dict[emotion] = features_avg

print('stop')