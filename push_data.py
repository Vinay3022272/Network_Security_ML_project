import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGODB_URL=os.getenv("MONGO_DB_URL")

import certifi
ca = certifi.where()

import numpy as np
import pandas as pd
import pymongo
from src.exception.exception import NetWorkSecurityException
from src.logging.logger import logging

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetWorkSecurityException(e, sys)
    
    def csv_to_json_converter(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            # converted data(json data) to upload in mongodb
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetWorkSecurityException(e, sys)
        
    def insert_data_mongodb(self, records, database, collection):
        try:
            self.database = database
            self.collection = collection
            self.records = records
            
            self.mongo_client = pymongo.MongoClient(MONGODB_URL)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return (len(self.records))
        except Exception as e:
            raise NetWorkSecurityException(e, sys)
        
# if __name__=="__main__":
#     FILE_PATH = "network_data\phisingData.csv"
#     DATABASE = "Network_db"
#     collection = "NetworkData"
#     networkobj = NetworkDataExtract()            
#     records = networkobj.csv_to_json_converter(FILE_PATH)
#     print(records)
#     no_of_records = networkobj.insert_data_mongodb(records, DATABASE, collection)
#     print(no_of_records)