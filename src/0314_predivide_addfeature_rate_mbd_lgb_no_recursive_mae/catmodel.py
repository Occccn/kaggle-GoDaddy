import os

import pandas as pd
from catboost import CatBoostRegressor
import numpy as np
import optuna
from recursive_model import RecursiveModel


class CATModel(RecursiveModel):
    def __init__(self,cfg):
        self.LGBCFG        = cfg
        self.LOG_EVAL      = 100
        self.NOT_FEATURES  = ['cfips','first_day_of_month','target','shift_target','microbusiness_density', 'row_id']
        self.TARGET        = 'target'
        self.DROP_COUNTY   = []
        self.PARAMS        = self.LGBCFG['params']
    

    def train_model(self): 
        #SMAPEでearly stopをかけるように修正
        cat_model = CatBoostRegressor(
            iterations=2000,
            loss_function="MAPE",
            verbose=0,
            grow_policy='SymmetricTree',
            learning_rate=0.035,
            colsample_bylevel=0.8,
            max_depth=5,
            l2_leaf_reg=0.2,
            subsample=0.70,
            max_bin=4096,
        )
        cat_model.fit(self.mart_train.drop(columns = self.NOT_FEATURES), self.mart_train[self.TARGET])
        self.preds  = cat_model.predict(self.mart_val.drop(columns = self.NOT_FEATURES))
