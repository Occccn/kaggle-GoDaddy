# Learning.py

# /////////////////////////////////////
# ////// Import Library ////////////////
# /////////////////////////////////////
# Third Party
import pandas as pd
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
        elif (_cfips_config["model"]== "LinearRegressor_active"):
            self.model = LinearRegressor_active(_cfips_config, _metric)
        
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
    
# --- Predict  Active ---
class LinearRegressor_active(Model):
    def __init__(self, _cfips_config, _metric):
        super().__init__( _cfips_config, _metric) 
        self.model_api = LinearRegression()
        
    def set_data(self, _data, _train_index, _val_index, _submit_index):
        self.data = _data
        self.train_index = _train_index
        self.val_index = _val_index
        self.submit_index = _submit_index
        # あとでmicrobusiness_densityを予測できるようにcfips, yearごとに人口相当の値を付与
        self.data = self.data.copy()
        self.data = self.data.reset_index()
        self.data["number_of_people"] = self.data["active"] / self.data["microbusiness_density"]
        first_day_of_month = pd.to_datetime(self.data["first_day_of_month"])
        self.data["year"] = first_day_of_month.dt.year
        number_of_people_df = self.data.groupby(["cfips", "year"])["number_of_people"].agg("mean")
        self.data = pd.merge(self.data, number_of_people_df, on=["cfips", "year"], suffixes=["", "_peopledf"])
        
        self.data = self.data.set_index("index")
        train = self.data.query("index in @_train_index")
        val = self.data.query("index in @_val_index")
        sub = self.data.query("index in @_submit_index")
        self.train_y = train["active"].values
        # 評価時はmicrobusiness_densityのため、val, subはmicrobusiness_densityにする
        self.val_y = val["microbusiness_density"].values
        self.sub_y = sub["microbusiness_density"].values
        self.train_x = np.arange(len(train)).reshape((-1, 1))
        self.val_x = np.arange(1 + len(val)).reshape((-1, 1))
        self.sub_x = np.arange(len(self.val_x) + len(sub)).reshape((-1, 1))

        
    def run(self):
        # 学習の実行
        self.model_api.fit(self.train_x, self.train_y)
        # Predict
        predict = self.model_api.predict(self.sub_x)
        shift = self.train_y[-1] - predict[0]
        self.predict_val = predict[1:len(self.val_x)] + shift
        self.predict_sub = predict[len(self.val_x):len(self.sub_x)] + shift
        
        # Post処理（active -> microbusiness_density)
        ## val
        for i, index in enumerate(self.val_index):
            tmp_year = self.data.iloc[index - self.val_index[0], self.data.columns.get_loc("year")]
            # 現状2023年のデータが無いので、2023年のデータは2022年と同じもので対応
            if tmp_year == 2023:
                tmp_year = 2022
            number_of_people = np.array(self.data.loc[self.data["year"]==tmp_year, "number_of_people"])[0] # 0を参照しているのは、同じ年であればいつでも同じため　とりあえず０
            self.predict_val[i] /= number_of_people
        ## sub
        for i, index in enumerate(self.submit_index):
            tmp_year = self.data.iloc[index - self.submit_index[0], self.data.columns.get_loc("year")]
            # 現状2023年のデータが無いので、2023年のデータは2022年と同じもので対応
            if tmp_year == 2023:
                tmp_year = 2022
            number_of_people = np.array(self.data.loc[self.data["year"]==tmp_year, "number_of_people"])[0] # 0を参照しているのは、同じ年であればいつでも同じため　とりあえず０
            self.predict_sub[i] /= number_of_people
        # metric
        self.metric_val = self.metric(self.predict_val, self.val_y)
        self.metric_sub = self.metric(self.predict_sub, self.sub_y)
        
    def predict(self, _input): # 使わないかもだけど
        predict = self.model_api.predict(_input)
        return predict
