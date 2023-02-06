# train.py

# --- Import Library ---
# Standard
import json
import os
import sys
# Third Party
import numpy as np
import pandas as pd
from tqdm import tqdm
# Original
from Learning import Learning
from CV import CV
# 想定環境
# latest
# |
# |- train.py
# |- Leaning.py
# |- CV.py

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

# --- Main ---
#群ごとに予測を実施
cfips_list = []
loss_list = []
inference_dict ={}
for cfips in tqdm(config.keys()):
    # 郡ごとにデータの抽出
    cfips_config = config[cfips]
    train_county = train.loc[train["cfips"]==int(cfips), :]
    test_county = test.loc[test["cfips"]==int(cfips), :]
    # クラスの初期化
    cv = CV(cv_type, train_county)
    learning = Learning(cfips_config, metric)
    
    # Divide Data For Cross Validation
    # Note: CVをループに回すのは、一旦保留。
    train_index, val_index, submit_index = cv.divide_data()
    # Learning & Predict
    learning.set_data(train_county, train_index, val_index)
    learning.run()
    metric_result = learning.get_metric_result()
    predict = learning.get_predict_val()

    # Data Storage
    cfips_list.append(cfips)
    loss_list.append(metric_result)
    inference_dict[cfips] = predict
    

# --- Post ---
# loss.csv
with open(loss_filepath, mode="w") as f:
    f.write("cfips, Loss\n")
    for cfips, loss in zip(cfips_list, loss_list):
        f.write(f"{cfips}, {loss}\n")
# predict.csv (いいやり方ではないと思う。。。とりあえずバリデーション期間が変わってもできるように)
inference_df = pd.DataFrame.from_dict(inference_dict)
inference_df.to_csv(inferenced_filepath)