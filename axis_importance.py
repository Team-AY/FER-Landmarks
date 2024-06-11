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

# get all of the emotions
emotions = np.unique(Train_labels)

# init dictionary to save all of the mean and std for each combination of emotion and landmark point
emotions_dict = {}

# iterate through all the emotions
# and for each combination of emotion and landmark point
# calculate the mean and std values 
for emotion in emotions:
    
    # positions of all the examples that match the emotion
    condition_df = Train_labels==emotion

    # all the examples that match the emotion
    data = Train_data[condition_df[0]]

    features_mean = []
    features_std = []

    # iterate through all the landmark points
    for i in range(68):
        test = np.array(data[i].values)
        a = [*zip(*test)]
        b = [list(i) for i in zip(*test)]
        test1 = np.array(b)
        features_mean.append(test1.mean(1))
        features_std.append(test1.std(1))

    emotions_dict[emotion] = {'mean': features_mean, 'std': features_std}

print('stop')