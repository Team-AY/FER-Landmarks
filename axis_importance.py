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

from matplotlib.patches import Ellipse


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

    # init the mean and std of the features for the current emotion
    features_mean = []
    features_std = []

    # iterate through all the landmark points
    for i in range(68):

        # turn the pandas into numpy array inorder to access as tuple the values
        test = np.array(data[i].values)

        # convert string to tuple
        a = [*zip(*test)]
        b = [list(i) for i in zip(*test)]
        test1 = np.array(b)

        # calculate mean and std for the current emotion and landmark point combination
        features_mean.append(test1.mean(1))
        features_std.append(test1.std(1))

    # convert the list of mean and std of the features into a numpy array
    features_mean = np.array(features_mean)
    features_std = np.array(features_std)

    # save the results of the mean and std for the current emotion
    emotions_dict[emotion] = {'mean': features_mean, 'std': features_std}

# unique list of colors for each emotion
color_list = ['lightcoral', 'lime', 'navy', 'purple', 'olive', 'skyblue', 'darkslategray']

# iterate over landmarks
for i in range(68):

    # create figure for each landmark
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    # init holder vectors
    x_vec = []
    y_vec = []
    width_vec = []
    height_vec = []

    # iterate over the emotions
    for j,emotion in enumerate(emotions):    

        # create ellipsoid for each emotion
        e = Ellipse(xy=emotions_dict[emotion]['mean'][i],
                    width=emotions_dict[emotion]['std'][i][0],
                    height=emotions_dict[emotion]['std'][i][1])
        
        # add the ellipsoid to the figure
        ax.add_artist(e)    

        # add centroid of the ellipsoid
        plt.scatter(emotions_dict[emotion]['mean'][i,0], emotions_dict[emotion]['mean'][i,1], c=color_list[j], label=emotion)
        
        # ellipsoid styling
        e.set_edgecolor(color_list[j])
        e.set_fill(False)

        # insert the values into the vector
        x_vec.append(emotions_dict[emotion]['mean'][i][0])
        y_vec.append(emotions_dict[emotion]['mean'][i][1])
        width_vec.append(emotions_dict[emotion]['std'][i][0])
        height_vec.append(emotions_dict[emotion]['std'][i][1])

    # show the legend
    fig.legend() 

    # adjust the axis lim
    ax.set_xlim(min(x_vec)-max(width_vec), max(x_vec)+max(width_vec))
    ax.set_ylim(min(y_vec)-max(height_vec), max(y_vec)+max(height_vec))

    # invert y as the data is from an image
    ax.invert_yaxis()

    # title the figure
    plt.title(f'Landmark #{i+1}')

    # save the figure
    fig.savefig(f'reports/axis_importance/landmark_{i+1}.png')       