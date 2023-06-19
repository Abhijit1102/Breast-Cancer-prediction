import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from src.logger import logging  # Update with the correct import statement
from src.exception import CustomException  # Update with the correct import statement
from src.utils import save_object  # Update with the correct import statement
from src.utils import evaluate_model  # Update with the correct import statement
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    train_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info('Splitting dependent and independent variables')
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            model_params = {
                'C': 0.1,
                'l1_ratio': None,
                'max_iter': 100,
                'n_jobs': -1,
                'penalty': 'l2',
                'solver': 'liblinear'
            }

            models = {
                "Logistic Regression": LogisticRegression(**model_params),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "SVM": SVC(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "MLP": MLPClassifier(),
                "XGBoost": XGBClassifier(),
                "LightGBM": LGBMClassifier(),
                "CatBoost": CatBoostClassifier(verbose=False)
            }

            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)  # Implement evaluate_model function

            print("=" * 35)
            logging.info(f"Model Report: {model_report}")

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name: {best_model_name}, Accuracy: {best_model_score}")
            print("=" * 35)
            logging.info(f"Best Model Found, Model Name: {best_model_name}, Accuracy: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

        except Exception as e:
            raise CustomException(e, sys)

