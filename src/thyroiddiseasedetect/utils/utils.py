import os
# smita - add path to parent folder to access other folder files 
import sys
import pickle 
from pathlib import Path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
print(sys.path)
#smita
from thyroiddiseasedetect.my_logging import logging
from thyroiddiseasedetect.exception import customexception
from sklearn.metrics import accuracy_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path) ##It extracts the directory path from the file_path

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train, y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            # get accuracy score
            test_accuracy = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_accuracy 

        return report

    except Exception as e:
        logging.info('Exception occurred during model training')
        raise customexception(e, sys)

        
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occurred in load_object function utils')
        raise customexception(e, sys)
