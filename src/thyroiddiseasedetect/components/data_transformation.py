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

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from thyroiddiseasedetect.my_logging import logging
from thyroiddiseasedetect.exception import customexception
from thyroiddiseasedetect.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info('Data Transformation initiated')
            numerical_columns = ['TSH', 'T3', 'TT4', 'FTI']
            categorical_columns = ['sex', 'on thyroxine', 'query hypothyroid', 'psych', 'TSH measured', 'pregnant']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise customexception(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            # Read data 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            

            preprocessing_obj = self.get_data_transformer_object()

            outcome = "outcome"

            # Drop specified columns
            drop_columns = [
                "goitre", "referral source", "on antithyroid medication", "thyroid surgery",
                "T3 measured", "TT4 measured", "query hyperthyroid", "age", "query on thyroxine",
                "lithium", "T4U measured", "T4U", "FTI measured", "hypopituitary", "tumor",
                "I131 treatment", "sick", "TBG measured", "TBG","outcome"
            ]

            ## X train 
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1) ## X train 

            
            ## target col  y train
            # Encode the target column
            label_encoder = LabelEncoder()
            target_feature_train_df = label_encoder.fit_transform(train_df[outcome])  # y train


            ## X test 
            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
           
            # Encode the target column for test data
            label_encoder = LabelEncoder()
            target_feature_test_df = label_encoder.fit_transform(test_df[outcome])  # y test

            # Convert NumPy arrays back to Pandas DataFrames
            input_feature_train_df = pd.DataFrame(input_feature_train_df, columns=input_feature_train_df.columns)
            input_feature_test_df = pd.DataFrame(input_feature_test_df, columns=input_feature_test_df.columns)

            print(input_feature_train_df.head())

            # Apply preprocessing object on training and testing datasets (Xtrain)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            

            logging.info("Applying preprocessing object on training and testing datasets.")

            # Handle imbalanced data using SMOTE
          
            smote = SMOTE(random_state=11, n_jobs=1)


            X_smote, y_smote = smote.fit_resample(input_feature_train_arr, target_feature_train_df)



            train_arr = np.c_[X_smote, y_smote]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            # Save preprocessing object
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj)

            

            logging.info("Preprocessing pickle files saved")

            return train_arr, test_arr
        except Exception as e:
            logging.exception("Exception occurred in the initialize_data_transformation")
            raise customexception(e, sys)
