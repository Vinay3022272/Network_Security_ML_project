from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.exception.exception import NetWorkSecurityException
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.entity.config_entity import TrainingPipelineConfig
import sys,os

if __name__=="__main__":
    try:
        
        training_pipeline_config = TrainingPipelineConfig() #automaticaly it will take timestamps
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiate the data ingestion")
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data initiation completed")
        print(data_ingestion_artifact)
        
        logging.info("Initiated the Data validation ")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
        print(data_validation_artifact)
        logging.info("data validation completed")
        
    except Exception as e:
        raise NetWorkSecurityException(e, sys)