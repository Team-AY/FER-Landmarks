import pandas as pd 
import sklearn
import math

import sklearn.utils
from model import Model
import matplotlib.pyplot as plt
import pickle
import numpy as np
import landmarks_annot as lm
from datetime import datetime
import os


#Prepare Data

#scale the data for normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#load the train data as dataframe
Train_data = pd.read_csv("datasets/landmarks/landmarks_relative_480x480_XY_Concat_features_train.csv" , header=None)
Train_data = Train_data.values #get values

#load the train labels as dataframe
Train_labels = pd.read_csv("datasets/landmarks/landmarks_relative_480x480_XY_Concat_labels_train.csv", header=None)
Train_labels = Train_labels.values.ravel() #set the right shape

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Train_labels = le.fit_transform(Train_labels)

#load the test data as dataframe and scale the data for normalization
Test_data = pd.read_csv("datasets/landmarks/landmarks_relative_480x480_XY_Concat_features_test.csv" , header=None)
Test_data = Test_data.values

#load the test labels as dataframe
Test_labels = pd.read_csv("datasets/landmarks/landmarks_relative_480x480_XY_Concat_labels_test.csv", header=None)
Test_labels = Test_labels.values.ravel()

Test_labels = le.transform(Test_labels)

#scaler = scaler.fit(np.concatenate((Train_data, Test_data)))
scaler = scaler.fit(Train_data)
Train_data = scaler.transform(Train_data)
Test_data = scaler.transform(Test_data)

# rename
X_train, X_test, y_train, y_test = Train_data, Test_data, Train_labels, Test_labels

# shuffle data
X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state=10)

# remove disugst from dataset
# X_train = X_train[y_train!='disgust']
# y_train = y_train[y_train!='disgust']
# X_test = X_test[y_test!='disgust']
# y_test = y_test[y_test!='disgust']

#from imblearn.under_sampling import RandomUnderSampler
#rus = RandomUnderSampler(random_state=0)
#X_train, y_train = rus.fit_resample(X_train, y_train)
#X_test, y_test = rus.fit_resample(X_test, y_test)

#from imblearn.over_sampling import SMOTE
#smote = SMOTE(random_state=0)
#X_train, y_train = smote.fit_resample(X_train, y_train)




# focus on specific facial features
#facial_features = lm.feature_selection(lm.ALL)
#X_train = X_train[:,facial_features]
#X_test = X_test[:,facial_features]

model_params = {'KNN': {},
                'QDA': {},
                'LDA': {},
                'GNB': {},
                'DT': {'random_state': 0},
                #'DT': {'max_depth': 9, 'max_features': 136, 'max_leaf_nodes': 370,
                #       'min_samples_leaf': 45, 'min_samples_split': 202,  'random_state': 0, 'class_weight': 'balanced'},
                #'DT': {'max_depth': 29, 'max_features': 127, 'max_leaf_nodes': 410, 
                #       'random_state': 0, 'class_weight': 'balanced'},
                'RF': {'random_state': 0},
                'XGBoost': {'random_state': 0}}

eval_result = {}
fitted_classifiers = {}
for (model_name, model_params) in model_params.items():
    print(f'Model Name: {model_name}')
    print(f'Model Parameters: {model_params}')
    clf = Model(model_name, model_params)
    (fitted_classifiers[model_name], eval_result[model_name]) = clf.eval_clf(X_train, X_test, y_train, y_test)


current_datetime = datetime.today().strftime('%Y%m%d%H%M%S')   
folder_name =  f'models/classifiers/relative_XY_Concat_{current_datetime}'

if not os.path.exists(folder_name):       
    os.makedirs(folder_name) 
    
os.path.join(folder_name, 'eval_result.pkl')
with open(os.path.join(folder_name, 'eval_result.pkl'), 'wb') as f:
  # dump information to that file
  pickle.dump(eval_result, f)

df = pd.DataFrame(eval_result)
df.to_csv(os.path.join(folder_name, 'eval_result.csv'))  

with open(os.path.join(folder_name, 'fitted_classifiers.pkl'), 'wb') as f:
  # dump information to that file
  pickle.dump(fitted_classifiers, f)

with open(os.path.join(folder_name, 'scaler.pkl'), 'wb') as f:
  # dump information to that file
  pickle.dump(scaler, f) 

with open(os.path.join(folder_name, 'label_encoder.pkl'), 'wb') as f:
  # dump information to that file
  pickle.dump(le, f)   

print("stop")