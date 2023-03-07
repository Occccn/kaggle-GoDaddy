import pandas as pd
import lightgbm as lgb
import numpy as np
from no_recursive_model import NoRecursiveModel

class LGBModel(NoRecursiveModel):
    def train_model(self): 
        #SMAPEでearly stopをかけるように修正
        dtrain      = lgb.Dataset(self.mart_train.drop(columns = self.NOT_FEATURES), self.mart_train[self.TARGET])
        dvalid      = lgb.Dataset(self.mart_val.drop(columns = self.NOT_FEATURES), self.mart_val[self.TARGET])
        gbm         = lgb.train(self.PARAMS, dtrain, valid_sets=[dvalid],callbacks=[lgb.log_evaluation(self.LOG_EVAL)])
        self.preds  = gbm.predict(self.mart_val.drop(columns = self.NOT_FEATURES))

