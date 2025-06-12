import os
import sys
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts',"train.csv")
    test_data_path:str=os.path.join('artifacts',"test.csv")
    raw_data_path:str=os.path.join('artifacts',"data.csv")

class DataIngestion:

    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def Initiate_data_ingestion(self):
        logging.info('data ingestion method or compoenent')
        try:
            df = pd.read_csv(os.path.join("Notebook", "Data", "stud.csv"))
            logging.info(f"Looking for file at: {os.path.abspath(os.path.join('Notebook', 'Data', 'stud.csv'))}")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            logging.info(f"Creating folder: {os.path.dirname(self.ingestion_config.train_data_path)}")
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('train split initiative')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('ingestion of the data is completed')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ =='__main__':
    obj=DataIngestion()
    obj.Initiate_data_ingestion()


