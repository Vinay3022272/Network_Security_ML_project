from src.exception.exception import NetWorkSecurityException
from src.logging.logger import logging

# configutaion of data ingestion config
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import pymongo
import pandas as pd
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

MONGODB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetWorkSecurityException(e, sys)
        
    def export_collection_as_dataframe(self):
        '''Reading data from mongodb and converted to dataframe'''
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            
            mongo_client = pymongo.MongoClient(MONGODB_URL)
            collection = mongo_client[database_name][collection_name]
            
            df=pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"])
                
            df.replace({"na": np.nan}, inplace=True)
            logging.info("data fetched successfully and converted to df")
            return df    
        
        except Exception as e:
            raise NetWorkSecurityException(e, sys)    
  
    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            # creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logging.info("data successfully saved to the feature store")
            return dataframe
        except Exception as e:
            raise NetWorkSecurityException(e, sys)    
        
    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed tarin test split on the dataframe")
            logging.info("Exited split_data_as_train_test method of Data_ingestion class")
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info("Exported train and test file path")
            
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            
            logging.info(f"Exported train and test file path")
            
        except Exception as e:
            raise NetWorkSecurityException(e, sys)        
        
    def initiate_data_ingestion(self):
        try:
            df = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(df)
            self.split_data_as_train_test(dataframe)
            data_ingestion_artifact = DataIngestionArtifact(
                self.data_ingestion_config.training_file_path,
                self.data_ingestion_config.testing_file_path
                )
            return data_ingestion_artifact
        except Exception as e:
            raise NetWorkSecurityException(e, sys)    