import pandas as pd
from catboost import CatBoostRegressor
import numpy as np
from no_recursive_model import NoRecursiveModel


class CATModel(NoRecursiveModel):

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
