from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os, sys
from import_data import DataImporter
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation initiated")
            
            data_importer = DataImporter()
            X, _ = data_importer.import_data()

            num_features = X.select_dtypes(exclude="object").columns
            cat_features = X.select_dtypes(include="object").columns
            
            logging.info("Data Transformation pipeline initiated")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, cat_features),
                    ("StandardScaler", numeric_transformer, num_features),        
                ]
            )
            
            logging.info("Data Transformation Completed")
            return preprocessor

        except Exception as e:
            logging.info('Exception Occurred in Data Transformation')
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            logging.info(f"Train Dataframe Head:\n{train_df.head().to_string()}")

            test_df = pd.read_csv(test_data_path)
            logging.info(f"Test Dataframe Head:\n{test_df.head().to_string()}")

            logging.info("Reading train and test data completed")

            preprocessing_obj = self.get_data_transformation_object()

            target_column = 'target'
            drop_columns = [target_column]

            input_feature_train = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column]

            ## Data Transformation

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessed obj on Train and test set")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
