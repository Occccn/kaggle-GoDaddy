import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os
import yaml
from tqdm import tqdm

from lgbmodel_no_recursive import LGBModel

# tmp
import warnings
warnings.simplefilter('ignore')

# --- Define Function ---
# ----- Metric -----
# Copy and Paste
# https://www.kaggle.com/code/cdeotte/seasonal-model-with-validation-lb-1-091
def smape(y_pred, y_true):
    # CONVERT TO NUMPY
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # WHEN BOTH EQUAL ZERO, METRIC IS ZERO
    both = np.abs(y_true) + np.abs(y_pred)
    idx = np.where(both==0)[0]
    y_true[idx]=1; y_pred[idx]=1
    
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
# -------------------------


# --- Set Paramters ---
# *********
n_trials = 50
# **********
data_folder = "../../data/raw"
train_filepath = os.path.join(data_folder, "train.csv")
test_filepath = os.path.join(data_folder, "test.csv")
revealed_test_filepath = os.path.join(data_folder, "revealed_test.csv")
census_starter_filepath = os.path.join(data_folder, "census_starter.csv")
submission_filepath = os.path.join(data_folder, "sample_submission.csv")
config_filepath = "./config.json"
save_dirpath = "./train"
loss_filepath = os.path.join(save_dirpath, "loss.csv")
inferenced_filepath = os.path.join(save_dirpath, "inferenced.csv")
result_optuna_param_filepath = "./optuna_result_param.csv"

# --- Preparation ---
# 変数の定義
train           = pd.read_csv(train_filepath)
test            = pd.read_csv(test_filepath)
revealed_test   = pd.read_csv(revealed_test_filepath)
census_starter  = pd.read_csv(census_starter_filepath)
submission      = pd.read_csv(submission_filepath)
# 出力ファイルが有れば削除
if os.path.exists(result_optuna_param_filepath):
    os.remove(result_optuna_param_filepath)
if os.path.exists("./result_optuna.csv"):
    os.remove("./result_optuna.csv")


train = pd.concat([train, revealed_test])
test  = pd.merge(test,train[['cfips', 'county', 'state']].drop_duplicates(),how = 'left')
test  = test[~test['first_day_of_month'].isin(['2022-11-01','2022-12-01'])]
train = pd.concat([train,test]).reset_index(drop = True)

# 設定ファイル関連
with open(config_filepath, mode="rt", encoding="utf-8") as f:
	config = json.load(f)
# CVのやり方（とりあえず別途管理するまでは、Noneにする）
cv_type  = None # cv_type = config["CV"]
# 評価方法
metric = smape
# 結果保存場所の環境作成
os.makedirs(save_dirpath, exist_ok=True)

# lgb用設定ファイルを読み込み
with open('lgbconfig.yml', 'r') as yml:
    LGBCFG = yaml.load(yml, Loader=yaml.SafeLoader)
    
model = LGBModel(LGBCFG)
model.set_data(train)
best_param, tmp_trial_result = model.tuning(n_trials)

# Post
param_result_dict = {}
for key in tmp_trial_result[0].params.keys():
    param_result_dict[key] = []
for i in range(len(tmp_trial_result)):
    trial_result_i = tmp_trial_result[i]
    for key, value in trial_result_i.params.items():
        param_result_dict[key].append(value)
param_result_df = pd.DataFrame().from_dict(param_result_dict)
param_result_df.to_csv(result_optuna_param_filepath)
   
