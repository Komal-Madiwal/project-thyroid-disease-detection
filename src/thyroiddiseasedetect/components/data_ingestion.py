import os
# smita - add path to parent folder to access other folder files 
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
print(sys.path)
import pandas as pd
import numpy as np

#smita
from thyroiddiseasedetect.my_logging import logging
from thyroiddiseasedetect.exception import customexception
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    #root_dir = "E:/ineuron_2023/thyroiddieaseprediction/thyroid_disease_detection"
    train_data_path = r"E:\ineuron_2023\thyroiddieaseprediction\thyroid_disease_detection\artifacts\train.csv"
    test_data_path = r"E:\ineuron_2023\thyroiddieaseprediction\thyroid_disease_detection\artifacts\test.csv"
    raw_data_path = r"E:\ineuron_2023\thyroiddieaseprediction\thyroid_disease_detection\artifacts\raw.csv"
    # train_data_path = os.path.join(root_dir, "artifacts", "train.csv")
    # test_data_path = os.path.join(root_dir, "artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            data_thyroid = pd.read_csv(os.path.join("notebook", "data", "hypothyroid.csv"))
            logging.info("Read dataset as a dataframe")

            # List of columns to convert to numeric
            numeric_columns = ['TSH', 'T3', 'TT4', 'FTI']  # Replace with your actual column names

            # Convert specified columns to numeric data types
            data_thyroid[numeric_columns] = data_thyroid[numeric_columns].apply(pd.to_numeric, errors='coerce')


            ## replacing '?' with Nan

            data_thyroid.replace('?',np.nan,inplace=True)
            data_thyroid

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Replace binary class column name with "outcome"
            binary_class_column = "binaryClass"  # Replace this with your actual binary class column name
            if binary_class_column in data_thyroid.columns:
                data_thyroid = data_thyroid.rename(columns={binary_class_column: "outcome"})
                data_thyroid.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Saved the raw dataset in the artifacts folder")
            logging.info("Performed train-test split")

            train_data, test_data = train_test_split(data_thyroid, test_size=0.25)

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed")
            
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.info("Exception occurred at data ingestion stage")
            raise customexception(e, sys)
        

## i have written this code in train_pipeline.py file

# if __name__ == "__main__":
#     obj = DataIngestion()
#     train_data, test_data = obj.initiate_data_ingestion()
