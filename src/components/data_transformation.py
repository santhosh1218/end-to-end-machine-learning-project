import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DatatransformationConfig:
    processor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class Datatransformation:
    def __init__(self):
        self.data_transformation_config=DatatransformationConfig()

    def get_data_transformer_object(self):# this funciton is create pikcle file where we do feature transformation
        try:
            numerical_columns=['writing_score','reading_score']
            categerical_columns=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            num_pipeline=Pipeline(

                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ])
            logging.info(f"numerical columns:{numerical_columns}")
            logging.info(f"categerical columns:{categerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ('numerical_pipeline',num_pipeline,numerical_columns),
                    ('catagerical_pipeline',cat_pipeline,categerical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def Initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('read train and test is complted')
            logging.info('obtaing preprocessor object')
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name='math_score'
            numerical_columns=['writing_score','reading_score']
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info('applying preprocessing object on training and testing data')
            logging.info(f'shape of training  data {input_feature_train_df.shape},shape of the testing data {input_feature_test_df.shape}')


            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]



            logging.info('saving the preprocesser object')
 
            save_object(
                file_path=self.data_transformation_config.processor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.processor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)


