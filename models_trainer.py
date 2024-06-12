import pandas as pd 
import sklearn
import math

import sklearn.utils
from model import Model
import matplotlib.pyplot as plt
import pickle
import numpy as np
import landmarks_annot as lm


#Prepare Data

#scale the data for normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#load the train data as dataframe
Train_data = pd.read_csv("datasets/landmarks/landmarks_features_train.csv" , header=None)
Train_data = Train_data.values #get values

#load the train labels as dataframe
Train_labels = pd.read_csv("datasets/landmarks/landmarks_labels_train.csv", header=None)
Train_labels = Train_labels.values.ravel() #set the right shape

#load the test data as dataframe and scale the data for normalization
Test_data = pd.read_csv("datasets/landmarks/landmarks_features_test.csv" , header=None)
Test_data = Test_data.values

#load the test labels as dataframe
Test_labels = pd.read_csv("datasets/landmarks/landmarks_labels_test.csv", header=None)
Test_labels = Test_labels.values.ravel()

#scaler = scaler.fit(np.concatenate((Train_data, Test_data)))
scaler = scaler.fit(Train_data)
Train_data = scaler.transform(Train_data)
Test_data = scaler.transform(Test_data)

# rename
X_train, X_test, y_train, y_test = Train_data, Test_data, Train_labels, Test_labels

# shuffle data
X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state=10)

# focus on specific facial features
facial_features = lm.feature_selection(lm.ALL)
X_train = X_train[:,facial_features]
X_test = X_test[:,facial_features]

model_params = {'KNN': {},
                'QDA': {},
                'LDA': {},
                'GNB': {},
                'DT': {'random_state': 42},
                'RF': {'random_state': 42}}

eval_result = {}
fitted_classifiers = {}
for (model_name, model_params) in model_params.items():
    print(f'Model Name: {model_name}')
    print(f'Model Parameters: {model_params}')
    clf = Model(model_name, model_params)
    (fitted_classifiers[model_name], eval_result[model_name]) = clf.eval_clf(X_train, X_test, y_train, y_test)
    

with open('models/classifiers/eval_result.pkl', 'wb') as f:
  # dump information to that file
  pickle.dump(eval_result, f)

with open('models/classifiers/fitted_classifiers.pkl', 'wb') as f:
  # dump information to that file
  pickle.dump(fitted_classifiers, f)

with open('models/classifiers/scaler.pkl', 'wb') as f:
  # dump information to that file
  pickle.dump(scaler, f) 

print("stop")