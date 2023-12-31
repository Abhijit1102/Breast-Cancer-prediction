import os
import sys

from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from import_data import DataImporter  # Import the DataImporter class from data_importer.py

@dataclass
class DataIngestionconfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")

        try:
            data_importer = DataImporter()
            X, y = data_importer.import_data()  # Use the DataImporter instance to import data

            df = pd.concat([X, pd.Series(y, name='target')], axis=1)

            logging.info("Pandas DataFrame is Created")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Raw data is created")

            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)
            
            logging.info("DataFrame is splited into training set and test set")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occurred at data ingestion stage')
            raise CustomException(e, sys)
