import os
import sys
import pickle

import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info(f"Exception occurred in load_object function utils for path {file_path}")   
        raise CustomException(e, sys)  


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)  # Train model
            y_test_pred = model.predict(X_test)
            test_model_score = accuracy_score(y_test, y_test_pred)
            report[model_name] = test_model_score

        return report

    except Exception as e:
        logging.info("Exception raised while training models")
        raise CustomException(e, sys)
        


def save_object(file_path, obj):
    try:
        logging.info(f"Save object {obj} initiated")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
        logging.info(f"Exception occurred in save_object utils for {obj}")
