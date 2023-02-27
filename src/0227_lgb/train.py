# train.py

# --- Import Library ---
# Standard
import datetime
import json
import os
import shutil
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
# FilePath
data_folder = "../../data/raw"
train_filepath = os.path.join(data_folder, "train.csv")
test_filepath = os.path.join(data_folder, "test.csv")
census_starter_filepath = os.path.join(data_folder, "census_starter.csv")
submission_filepath = os.path.join(data_folder, "sample_submission.csv")
config_filepath = "./config.json"
save_dirpath = "./train"
loss_filepath = os.path.join(save_dirpath, "loss.csv")
inferenced_filepath = os.path.join(save_dirpath, "inferenced.csv")
# CV
## 全データを用いて学習を行う場合
cv_type  = None # cv_type = config["CV"]
## 直近のデータを用いて学習を行う場合
# cv_type = "from_202102"
# Metric
metric = smape
# Comment
comment = "operation_test"

# --- Preparation ---
# 変数の定義
train           = pd.read_csv(train_filepath)
test            = pd.read_csv(test_filepath)
census_starter  = pd.read_csv(census_starter_filepath)
submission      = pd.read_csv(submission_filepath)
# 設定ファイル関連
with open(config_filepath, mode="rt", encoding="utf-8") as f:
	config = json.load(f)
# 結果保存場所の環境作成
os.makedirs(save_dirpath, exist_ok=True)
# 結果のコピー先のファイルパス（検討結果とコード）
dt_now = datetime.datetime.now()
now_month, now_day = str(dt_now.month).zfill(2), str(dt_now.day).zfill(2)
now_hour, now_minute = str(dt_now.hour).zfill(2), str(dt_now.minute).zfill(2)
if cv_type == None:
    cv_type_comment = "None"
else:
    cv_type_comment = cv_type
cp_path = f"../{now_month}{now_day}_{now_hour}{now_minute}_cv_{cv_type_comment}_{comment}"
print(cp_path)

# --- Main ---
#群ごとに予測を実施
cfips_list = []
val_loss_list = []
sub_loss_list = []
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
    learning.model.set_data(train_county, train_index, val_index, submit_index)
    learning.model.run()
    metric_sub = learning.model.get_metric_sub()
    metric_val = learning.model.get_metric_val()
    predict_val = learning.model.get_predict_val()
    predict_sub = learning.model.get_predict_sub()
    predict = np.r_[predict_val, predict_sub]
    # Data Storage
    cfips_list.append(cfips)
    val_loss_list.append(metric_val)
    sub_loss_list.append(metric_sub)
    inference_dict[cfips] = predict
# --- Post ---
# loss.csv
with open(loss_filepath, mode="w") as f:
    f.write("cfips,val_loss,sub_loss\n")
    for cfips, val_loss, sub_loss in zip(cfips_list, val_loss_list, sub_loss_list):
        f.write(f"{cfips}, {val_loss}, {sub_loss}\n")
# predict.csv (いいやり方ではないと思う。。。とりあえずバリデーション期間が変わってもできるように)
inference_df = pd.DataFrame.from_dict(inference_dict)
inference_df.to_csv(inferenced_filepath)
# save result
shutil.copytree("../latest", cp_path)