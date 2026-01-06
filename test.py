from src.components.data_ingestion import DataIngestion
from src.exception.exception import NetWorkSecurityException
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import TrainingPipelineConfig
import sys,os
from src.utils.main_utils.utils import read_yaml_file

if __name__=="__main__":
    try:
        
        # training_pipeline_config = TrainingPipelineConfig() #automaticaly it will take timestamps
        # data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        # data_ingestion = DataIngestion(data_ingestion_config)
        # logging.info("Initiate the data ingestion")
        # data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
        # print(data_ingestion_artifact)
        
        SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")
        schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        print(len(schema_config["columns"]))

        
    except Exception as e:
        raise NetWorkSecurityException(e, sys)