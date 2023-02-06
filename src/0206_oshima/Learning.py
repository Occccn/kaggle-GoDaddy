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
        self.config = _cfips_config
        self.metric = _metric
        self.data = None
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.predict_val = None
        self.metric_result = None
        self.model_type = None
        self.model = None
        # self.model_param = None

        # About Model
        self.model_type = self.config["model"]
        ## Set Model（モデルごとのコンストラクタの作成）
        if (self.model_type == "LinearReression"):
            self.model = LinearRegression()
        else:
            pass
         
    def set_data(self, _data, _train_index, _val_index):
        self.data = _data
        
        train = self.data.query("index in @_train_index")
        val = self.data.query("index in @_val_index")
        
        self.train_y = train["microbusiness_density"].values
        self.val_y = val["microbusiness_density"].values
        if (self.model_type == "LinearReression"):
            self.train_x = np.arange(len(train)).reshape((-1, 1))
            self.val_x = np.arange(len(val)).reshape((-1, 1))
        else:
            pass

    def run(self):
        # 学習の実行
        # ここに分岐をつけて学習をそれぞれ学習
        if (self.model_type == "LinearReression"):
            self.model.fit(self.train_x, self.train_y)
            self.predict_val = self.model.predict(self.val_x)
            self.metric_result = self.metric(self.predict_val, self.val_y)
    
    def predict(self, _input): # 使わないかもだけど
        if (self.model_type == "LinearReression"):
            predict = self.model.predict(_input)
            return predict

    def get_model(self):
        return self.model
    
    def get_predict_val(self):
        return self.predict_val

    def get_metric_result(self):
        return self.metric_result

    # 以下、メソッドを追加する
    # ・学習時の過程を振り返られるようにLossを保存
    # ・学習時の推論結果を保存