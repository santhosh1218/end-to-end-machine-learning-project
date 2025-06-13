import os
import numpy as np
import pandas as pd
import sys
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score


def save_object(file_path,obj):
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def Evalute_model(x_train,y_train,x_test,y_test,models):
    try:
        if not isinstance(models,dict):
            raise CustomException('models must be in dictionary')
        report={}
        for name,model in models.items():
            model.fit(x_train,y_train)
            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)
            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)
            report[name]=test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)