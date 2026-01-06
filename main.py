from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.exception.exception import NetWorkSecurityException
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from src.entity.config_entity import TrainingPipelineConfig
import sys,os

if __name__=="__main__":
    try:
        
        training_pipeline_config = TrainingPipelineConfig() #automaticaly it will take timestamps
        
        logging.info("Initiate the data ingestion")
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        logging.info("Data initiation completed")
        
        logging.info("Initiated the Data validation ")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
        print(data_validation_artifact)
        logging.info("data validation completed")
        
        logging.info("Initiated Data Transformation")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("data transformation completed")
        
    except Exception as e:
        raise NetWorkSecurityException(e, sys)