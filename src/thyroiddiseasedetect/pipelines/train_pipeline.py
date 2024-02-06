import os
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
#print(sys.path)

from thyroiddiseasedetect.my_logging import logging
from thyroiddiseasedetect.exception import customexception
from thyroiddiseasedetect.components.data_transformation import DataTransformation
from thyroiddiseasedetect.components.data_ingestion import DataIngestion
from thyroiddiseasedetect.components.model_trainer import ModelTrainer
from thyroiddiseasedetect.utils.utils import save_object

# #creating object of DataIngestion class 
obj=DataIngestion()
train_data_path,test_data_path=obj.initiate_data_ingestion()

# # creating object of datatransformation class
if __name__ == "__main__":
    data_transformation = DataTransformation()

    # Call the initialize_data_transformation method and unpack the results
    train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path, test_data_path)


## creating ob for modeltrainer class
model_trainer_obj=ModelTrainer()
model_trainer_obj.initate_model_training(train_arr,test_arr)



