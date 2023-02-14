import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os

from lgbmodel import LGBModel

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
data_folder = "../../data/raw"
train_filepath = os.path.join(data_folder, "train.csv")
test_filepath = os.path.join(data_folder, "test.csv")
census_starter_filepath = os.path.join(data_folder, "census_starter.csv")
submission_filepath = os.path.join(data_folder, "sample_submission.csv")
config_filepath = "./config.json"
save_dirpath = "./train"
loss_filepath = os.path.join(save_dirpath, "loss.csv")
inferenced_filepath = os.path.join(save_dirpath, "inferenced.csv")

# --- Preparation ---
# 変数の定義
train           = pd.read_csv(train_filepath)
test            = pd.read_csv(test_filepath)
census_starter  = pd.read_csv(census_starter_filepath)
submission      = pd.read_csv(submission_filepath)
# 設定ファイル関連
with open(config_filepath, mode="rt", encoding="utf-8") as f:
	config = json.load(f)
# CVのやり方（とりあえず別途管理するまでは、Noneにする）
cv_type  = None # cv_type = config["CV"]
# 評価方法
metric = smape
# 結果保存場所の環境作成
os.makedirs(save_dirpath, exist_ok=True)





# --- Post ---
# loss.csv
tmp = []
tmp2 = []
for i in range(len(mart_val)):
    tmp2.append(mart_val['cfips'].iloc[i])
    loss = smape(np.array([mart_val['microbusiness_density'].iloc[i]]), 
                np.array([mart_val['preds_mbd'].iloc[i]]))
    tmp2.append(loss)
    tmp.append(tmp2)
    tmp2 = []
loss = pd.DataFrame(tmp)
loss.columns = ['cfips', 'val_loss']
loss.to_csv(loss_filepath, index = False)


# predict.csv (いいやり方ではないと思う。。。とりあえずバリデーション期間が変わってもできるように)
inference_df = mart_val[['preds_mbd']].T.reset_index(drop = True)
inference_df.columns = mart_val['cfips'].values
inference_df.to_csv(inferenced_filepath, index = False)