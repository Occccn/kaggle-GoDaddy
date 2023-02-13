# Learning.py

# /////////////////////////////////////
# ////// Import Library ////////////////
# /////////////////////////////////////
# Third Party
import numpy as np
# Scikit-learn
from sklearn.linear_model import LinearRegression
# Other

# /////////////////////////////////////
# ////// Define Class //////////////////
# /////////////////////////////////////

class Learning:
    def __init__(self, _cfips_config, _metric):
        if (_cfips_config["model"]== "LinearReression"):
            self.model = LinearRegressor(_cfips_config, _metric)
        elif (_cfips_config["model"]== "LastModel"):
            self.model = LastModel(_cfips_config, _metric)



class Model:
    def __init__(self, _cfips_config, _metric):
        self.config = _cfips_config
        self.metric = _metric
        self.data = None
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.sub_x = None
        self.sub_y = None
        self.predict_val = None
        self.metric_val = None
        self.modelinstance = None

    def set_data(self, _data, _train_index, _val_index, _submit_index):
        pass

    def run(self):
        # 学習の実行
        # ここに分岐をつけて学習をそれぞれ学習
        pass
    
    def predict(self, _input): # 使わないかもだけど
        pass

    def get_model(self):
        return self.model
    
    def get_predict_val(self):
        return self.predict_val
    
    def get_predict_sub(self):
        return self.predict_sub

    def get_metric_val(self):
        return self.metric_val
    
    def get_metric_sub(self):
        return self.metric_sub

    # 以下、メソッドを追加する
    # ・学習時の過程を振り返られるようにLossを保存
    # ・学習時の推論結果を保存
    
    
class LinearRegressor(Model):
    def __init__(self, _cfips_config, _metric):
        super().__init__( _cfips_config, _metric) 
        self.model_api = LinearRegression()
        
    def set_data(self, _data, _train_index, _val_index, _submit_index):
        self.data = _data
        
        train = self.data.query("index in @_train_index")
        val = self.data.query("index in @_val_index")
        sub = self.data.query("index in @_submit_index")
        self.train_y = train["microbusiness_density"].values
        self.val_y = val["microbusiness_density"].values
        self.sub_y = sub["microbusiness_density"].values
        self.train_x = np.arange(len(train)).reshape((-1, 1))
        self.val_x = np.arange(1 + len(val)).reshape((-1, 1))
        self.sub_x = np.arange(len(self.val_x) + len(sub)).reshape((-1, 1))

        
    def run(self):
        # 学習の実行
        # ここに分岐をつけて学習をそれぞれ学習
        self.model_api.fit(self.train_x, self.train_y)
        predict = self.model_api.predict(self.sub_x)
        shift = self.train_y[-1] - predict[0]
        self.predict_val = predict[1:len(self.val_x)] + shift
        self.predict_sub = predict[len(self.val_x):len(self.sub_x)] + shift
        self.metric_val = self.metric(self.predict_val, self.val_y)
        self.metric_sub = self.metric(self.predict_sub, self.sub_y)
    def predict(self, _input): # 使わないかもだけど
        predict = self.model_api.predict(_input)
        return predict
    
    
class LastModel(Model):
    def __init__(self, _cfips_config, _metric):
        super().__init__( _cfips_config, _metric) 
        
    def set_data(self, _data, _train_index, _val_index, _submit_index):
        self.data = _data
        
        train = self.data.query("index in @_train_index")
        val = self.data.query("index in @_val_index")
        sub = self.data.query("index in @_submit_index")
        self.train_y = train["microbusiness_density"].values
        self.val_y = val["microbusiness_density"].values
        self.sub_y = sub["microbusiness_density"].values
        self.train_x = np.arange(len(train)).reshape((-1, 1))
        self.val_x = np.arange(1 + len(val)).reshape((-1, 1))
        self.sub_x = np.arange(len(self.val_x) + len(sub)).reshape((-1, 1))

        
    def run(self):
        # 学習の実行
        # ここに分岐をつけて学習をそれぞれ学習

        lastval = self.train_y[-1]
        self.predict_val = [lastval for _ in range(len(self.val_y))]
        self.predict_sub = [lastval for _ in range(len(self.sub_y))]
        self.metric_val = self.metric(self.predict_val, self.val_y)
        self.metric_sub = self.metric(self.predict_sub, self.sub_y)
    def predict(self, _input): # 使わないかもだけど
        pass
    
    
