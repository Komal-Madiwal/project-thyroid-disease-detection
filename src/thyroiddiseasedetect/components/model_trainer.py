
import os
import warnings
import logging
from pathlib import Path
import sys

warnings.filterwarnings("ignore", category=UserWarning, module="joblib")


current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
print(sys.path)
import numpy as np
import pandas as pd
from dataclasses import dataclass
from thyroiddiseasedetect.my_logging import logging
from thyroiddiseasedetect.exception import customexception

from dataclasses import dataclass
from thyroiddiseasedetect.utils.utils import save_object
from thyroiddiseasedetect.utils.utils import evaluate_model

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,test_array): ## this is the output of data transformation 
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'Logistic Regression': LogisticRegression(),
            'SVM': SVC(),
            'Random Forest': RandomForestClassifier(),
            'Naive Bayes': GaussianNB(),
            'XGBoost': XGBClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'KNN': KNeighborsClassifier()
        }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models) ## from utlis 
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)

        
    