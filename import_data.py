import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd

from sklearn.datasets import load_breast_cancer

class DataImporter:
    def __init__(self):
        logging.info('Data importing initiated')

    def import_data(self):
        try:
            logging.info("Data import process starts.")

            # Load the breast cancer dataset from scikit-learn
            dataset = load_breast_cancer()

            # Create a pandas DataFrame from the dataset
            X = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
            y = dataset.target

            logging.info("Data import process completed.")
            return X, y

        except Exception as e:
            raise CustomException(e, sys)
