import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import Evalute_model

@dataclass
class ModerTrainingConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrianer:
    def __init__(self):
        self.model_trainer_config=ModerTrainingConfig()

    def Initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('split training and test split data')
            x_train,y_train,x_test,y_test=train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]

            models={
                'Linear_Regression':LinearRegression(),
                'KNN':KNeighborsRegressor(),
                'Decision_Tree':DecisionTreeRegressor(),
                'Random_Forest':RandomForestRegressor(),
                'Xgboost_Regressor':XGBRegressor(),
                'AdaBoost_Regressor':AdaBoostRegressor()
            }
            model_report:dict=Evalute_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            # to get the best score from the dictionary
            best_model_score=max(sorted((model_report.values())))
            # to get the best model from the dictionary
            best_model_name=max(model_report,key=model_report.get)
            # if best_model_score<0.6:
            #     raise CustomException('no best model found')
            best_model=models[best_model_name]
            logging.info(f'best model found on both training and testing')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            model_r2_score=r2_score(y_test,predicted)

            return model_r2_score

        except Exception as e:
            raise CustomException(e,sys)









