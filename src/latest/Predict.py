# predict.py


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
# Filepath
data_folder = "../../data/raw"
train_filepath = os.path.join(data_folder, "train.csv")
revealed_test_filepath = os.path.join(data_folder, "revealed_test.csv")
test_filepath = os.path.join(data_folder, "test.csv")
census_starter_filepath = os.path.join(data_folder, "census_starter.csv")
submission_filepath = os.path.join(data_folder, "sample_submission.csv")
config_filepath = "./config.json"
save_dirpath = "./train"
loss_filepath = os.path.join(save_dirpath, "loss.csv")
inferenced_filepath = os.path.join(save_dirpath, "inferenced.csv")
# Variable
train_cv_type = None
submit_cv_type = "submit_full_ver2"

# --- Preparation ---
# 変数の定義
train           = pd.read_csv(train_filepath)
revealed_test   = pd.read_csv(revealed_test_filepath)
test            = pd.read_csv(test_filepath)
census_starter  = pd.read_csv(census_starter_filepath)
submission      = pd.read_csv(submission_filepath)
# データの用意
## all data
df_list = []
for cfips in train["cfips"].unique():
    train_cfips_df           = train.loc[train["cfips"]==cfips, :]
    revealed_test_cfips_df   = revealed_test.loc[revealed_test["cfips"]==cfips, :]
    # testのデータは、2022/11月から始まっており、revealed_testと11,12月の2ヶ月重複があるため
    test_cfips_df            = test.loc[test["cfips"]==cfips, :][2:] 
    cfips_df = pd.concat([train_cfips_df, test_cfips_df])
    df_list.append(cfips_df)
all_data = pd.concat(df_list)
all_data.reset_index(drop=True, inplace=True)
## train
train = pd.concat([train, revealed_test])
train.reset_index(drop=True, inplace=True)
# 設定ファイル関連
with open(config_filepath, mode="rt", encoding="utf-8") as f:
	config = json.load(f)
# 評価方法
metric = smape
# 結果保存場所の環境作成
os.makedirs(save_dirpath, exist_ok=True)

# --- Main ---
#群ごとに予測を実施
# predict = np.zeros((all_data["cfips"].nunique()))
cfips_list = []
val_loss_list = []
sub_loss_list = []
inference_dict ={}
for i, cfips in tqdm(enumerate(config.keys())):
    # cfipsごとにデータの抽出
    cfips_config = config[cfips]
    # --- 既存のデータで学習モデル精度検証用の出力作成 ---
    # print("Learning For Validation ...")
    train_cfips_df = train.loc[train["cfips"]==int(cfips), :]
    cv = CV(train_cv_type, train_cfips_df)
    learning = Learning(cfips_config, metric)
    train_index, val_index, submit_index = cv.divide_data()
    learning.model.set_data(train_cfips_df, train_index, val_index, submit_index)
    learning.model.run()
    metric_sub = learning.model.get_metric_sub()
    metric_val = learning.model.get_metric_val()
    predict_val = learning.model.get_predict_val()
    predict_sub = learning.model.get_predict_sub()
    cfips_list.append(cfips)
    val_loss_list.append(metric_val)
    sub_loss_list.append(metric_sub)
    # print()
    
    # --- 提出用に再学習 ---
    # print("Learning For Submit ...")
    submit_cfips_df = all_data.loc[all_data["cfips"]==int(cfips), :]
    cv = CV(submit_cv_type, submit_cfips_df)
    learning = Learning(cfips_config, metric)
    train_index, val_index, submit_index = cv.divide_data()
    learning.model.set_data(submit_cfips_df, train_index, val_index, submit_index)
    learning.model.run()
    metric_sub = learning.model.get_metric_sub()
    metric_val = learning.model.get_metric_val()
    predict_val = learning.model.get_predict_val()
    predict_sub = learning.model.get_predict_sub()
    predict = np.r_[predict_val, predict_sub]
    inference_dict[cfips] = predict
    
    # # クラスの初期化
    # cv = CV(cv_type, df)
    # learning = Learning(cfips_config, metric)
    
    # # Divide Data For Cross Validation
    # # Note: CVをループに回すのは、一旦保留。
    # train_index, val_index, submit_index = cv.divide_data()
    # # Learning & Predict
    # learning.model.set_data(df, train_index, val_index, submit_index)
    # learning.model.run()
    # metric_sub = learning.model.get_metric_sub()
    # metric_val = learning.model.get_metric_val()
    # predict_val = learning.model.get_predict_val()
    # predict_sub = learning.model.get_predict_sub()
    # # predict = np.r_[predict_val, predict_sub]
    # predict[i] = predict_sub[1] # 11月のインデックスは、2番目のため（これで良いのか確認）
    
# --- Post ---
# target = pd.DataFrame(data={"microbusiness_density":predict}, index=all_data["cfips"].unique())
# submit = test.join(target, on="cfips")[["row_id", "microbusiness_density"]]
# submit.to_csv(f"{save_dirpath}/submit.csv", index=False)

# loss.csv
with open(loss_filepath, mode="w") as f:
    f.write("cfips,val_loss,sub_loss\n")
    for cfips, val_loss, sub_loss in zip(cfips_list, val_loss_list, sub_loss_list):
        f.write(f"{cfips}, {val_loss}, {sub_loss}\n")
# inference.csv
inference_df = pd.DataFrame.from_dict(inference_dict)
inference_df.to_csv(inferenced_filepath)
# save result
# shutil.copytree("../latest", cp_path)