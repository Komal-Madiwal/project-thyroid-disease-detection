
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

from src.thyroiddiseasedetect.exception  import customexception
from src.thyroiddiseasedetect.my_logging import logging
from src.thyroiddiseasedetect.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            scaled_data=preprocessor.transform(features)
            
            pred=model.predict(scaled_data)
            
            return pred
            
            
        
        except Exception as e:
            raise customexception(e,sys)
    
    
    
class CustomData:
    def __init__(self,
                 sex: str,
                 on_thyroxine: str,
                 pregnant: str,
                 query_hypothyroid: str,
                 psych: str,
                 TSH_measured: str,
                 TSH: float,
                 T3: float,
                 TT4: float,
                 FTI: float
                 ):
        self.sex = sex
        self.on_thyroxine = on_thyroxine
        self.pregnant = pregnant
        self.query_hypothyroid = query_hypothyroid
        self.psych = psych
        self.TSH_measured = TSH_measured
        self.TSH = TSH
        self.T3 = T3
        self.TT4 = TT4
        self.FTI = FTI

    def get_data_as_dataframe(self):
        try:
            # Create a dictionary with the provided attributes
            custom_data_input_dict = {
                'sex': [self.sex],
                'on thyroxine': [self.on_thyroxine],
                'pregnant': [self.pregnant],
                'query hypothyroid': [self.query_hypothyroid],
                'psych': [self.psych],
                'TSH measured': [self.TSH_measured],
                'TSH': [self.TSH],
                'T3': [self.T3],
                'TT4': [self.TT4],
                'FTI': [self.FTI]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')

            # Access columns with spaces using square brackets and strings
            query_hypothyroid_column = df['query hypothyroid']
            tsh_measured_column = df['TSH measured']

            return df
        except Exception as e:
            logging.info('Exception Occurred in prediction pipeline')
            raise customexception(e, sys)