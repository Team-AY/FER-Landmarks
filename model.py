from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import time
from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import numpy as np
import pandas as pd

class Model():
    def __init__(self, classifier_name: str, params):
        '''
        Input: classifier_name- represents the name of the classifier, load the matching classifier according to the name
               params- represents the parameters that will be sent to the classifier builder               
        Output: None
        Description: init the Model using the given classifier name
        '''
        if classifier_name == 'KNN':
            self.__clf__ = KNeighborsClassifier(**params)            
        elif classifier_name == 'QDA':
            self.__clf__ = QuadraticDiscriminantAnalysis(**params)
        elif classifier_name == 'LDA':
            self.__clf__ = LinearDiscriminantAnalysis(**params)
        elif classifier_name == 'GNB':
            self.__clf__ = GaussianNB(**params)
        elif classifier_name == 'DT':
            self.__clf__ = DecisionTreeClassifier(**params)          
        elif classifier_name == 'RF':
            self.__clf__ = RandomForestClassifier(**params)       
        elif classifier_name == 'XGBoost':
            self.__clf__ = XGBClassifier(**params)                                                               
        else:
            raise NameError(f'Classifier ({classifier_name}) Has Not Been Implemented.')

        
    def fit(self, X, y):
        '''
        Input: X- Features of the data
               y- Labels of the data
        Output: The fitted classifier
        Description: init the Model using the given classifier name
        '''        
        start = time.time()
        self.__clf__.fit(X,y)
        elapsed = time.time()-start
        print(f"Elapsed Fit Time: {elapsed:.3f} [s]")
    

    def predict(self, X):
        '''
        Input: X- Features of the data to be predicted               
        Output: The predictions for each example in the data
        Description: Predicts all of the examples in the data
        '''         
        start = time.time()                
        predictions = self.__clf__.predict(X)
        elapsed = time.time()-start
        print(f"Elapsed Prediction Time: {elapsed:.3f} [s]")
        return predictions   


    def score(self, X, y):
        '''
        Input: X- Features of the data
               y- Labels of the data
        Output: The mean accuracy on the given data and labels.
        Description: Returns the mean accuracy on the given data and labels.
        '''          
        start = time.time()                
        score = self.__clf__.score(X, y)
        elapsed = time.time()-start
        print(f"Elapsed Score Time: {elapsed:.3f} [s]")
        return score
    

    def eval_clf(self, X_train, X_test, y_train, y_test):  
        '''
        Input: X_train- Features of the train data
               X_test- Features of the test data
               y_train- Labels of the train data
               y_test- Labels of the test data
        Output: dictionary representing the evaluation result.
                where each key is an evaluation metric.
                each value in that is the result of an evaluation metric
                some values of evaluation matric can be a dictionary that contains the train and test results separately     
        Description: Evaluates the Model and returns the evaluation result.
        '''  
        # Start Evaluation Timer      
        start = time.time()
        print('Evaluation Started...')

        # fit the classifier
        self.fit(X_train,y_train)

        # predict on train and test
        prediction_train = self.predict(X_train)
        prediction_test = self.predict(X_test)

        # accuracy for train and test
        accuracy_train = self.score(X_train, y_train)
        accuracy_test = self.score(X_test, y_test)

        # confusion matrix for train and test
        cm_train = confusion_matrix(y_train, prediction_train, labels=np.unique(y_train), normalize= 'true')
        cm_test = confusion_matrix(y_test, prediction_test, labels=np.unique(y_test), normalize= 'true')

        # save the labels
        labels_train = np.unique(y_train)
        
        # take the train cofusion matrix and convert it to dataframe
        train_confMat_df = pd.DataFrame(cm_train)        

        # rename the columns and indexes to match the data labels
        train_confMat_df.columns = labels_train
        train_confMat_df.index = labels_train


        # save the labels
        labels_test = np.unique(y_test)
        
        # take the test cofusion matrix and convert it to dataframe
        test_confMat_df = pd.DataFrame(cm_test)        

        # rename the columns and indexes to match the data labels
        test_confMat_df.columns = labels_test
        test_confMat_df.index = labels_test

        # recall calculation
        recall_train = recall_score(y_train, prediction_train, average='weighted')
        recall_test = recall_score(y_test, prediction_test, average='weighted')

        # precision calculation
        precision_train = precision_score(y_train, prediction_train, average='weighted')
        precision_test = precision_score(y_test, prediction_test, average='weighted')    

        # f1 calculation
        f1_train = f1_score(y_train, prediction_train, average='weighted')
        f1_test = f1_score(y_test, prediction_test, average='weighted')              

        # End Evaluation Timer
        elapsed = time.time()-start        

        # Total Evaluation Time
        print(f"Elapsed Total Evaluation Time: {elapsed:.3f} [s]")

        # Evaluation has been completed
        print('Evaluation Completed!')

        # create evaluation result as a dictionary
        eval_result = {
                'accuracy': {'train': accuracy_train, 'test': accuracy_test}, 
                'cm': {'train': train_confMat_df, 'test': test_confMat_df},
                'recall': {'train': recall_train, 'test': recall_test},
                'precision': {'train': precision_train, 'test': precision_test},
                'f1': {'train': f1_train, 'test': f1_test}                
                }     
        
        return (self, eval_result)