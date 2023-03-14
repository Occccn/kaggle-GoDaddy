import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os
import yaml
from tqdm import tqdm

from lgbmodel_no_recursive import LGBModel

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
revealed_test_filepath = os.path.join(data_folder, "revealed_test.csv")
census_starter_filepath = os.path.join(data_folder, "census_starter.csv")
submission_filepath = os.path.join(data_folder, "sample_submission.csv")
config_filepath = "./config.json"
save_dirpath = "./train"
loss_filepath = os.path.join(save_dirpath, "loss.csv")
loss_last_filepath = os.path.join(save_dirpath, "loss_last.csv")
inferenced_filepath = os.path.join(save_dirpath, "inferenced.csv")

# --- Preparation ---
# 変数の定義
train           = pd.read_csv(train_filepath)
test            = pd.read_csv(test_filepath)
revealed_test   = pd.read_csv(revealed_test_filepath)
census_starter  = pd.read_csv(census_starter_filepath)
submission      = pd.read_csv(submission_filepath)

train = pd.concat([train, revealed_test])
test  = pd.merge(test,train[['cfips', 'county', 'state']].drop_duplicates(),how = 'left')
test  = test[~test['first_day_of_month'].isin(['2022-11-01','2022-12-01'])]
# === 加工 ===
divide_column_name = "rate_mbd"
divide_column_name_group = f"{divide_column_name}_groupby"
tmp = train.copy()
tmp.loc[tmp["microbusiness_density"]==0, "microbusiness_density"] = 1
tmp_shift1 = tmp.groupby("cfips")["microbusiness_density"].shift(1).fillna(method="bfill")
tmp[divide_column_name] = tmp["microbusiness_density"] / tmp_shift1
tmp_groupby = pd.DataFrame(tmp.groupby("cfips")[divide_column_name].mean())
tmp_groupby = tmp_groupby.reset_index()
tmp_groupby = tmp_groupby.rename(columns={divide_column_name:divide_column_name_group})
tmp = pd.merge(tmp, tmp_groupby, on="cfips")
print(tmp)
percentile_low = np.percentile(tmp[f"{divide_column_name}_groupby"], 33)
percentile_high = np.percentile(tmp[f"{divide_column_name}_groupby"], 66)
low = tmp.query("rate_mbd_groupby < @percentile_low")
low_cfips = low["cfips"].unique()
middle = tmp.query("@percentile_low <= rate_mbd_groupby < @percentile_high")
middle_cfips = middle["cfips"].unique()
high = tmp.query("@percentile_high <= rate_mbd_groupby")
high_cfips = high["cfips"].unique()
# ===
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
    

VAL_DATE = ['2022/8/1']
SUB_DATE = ['2022/9/1', '2022/10/1', '2022/11/1', '2022/12/1']
df_list = []
query_list = [low_cfips, middle_cfips, high_cfips]
for cfips_i in query_list:
    train_i = train.loc[train["cfips"].isin(cfips_i),:]
    model = LGBModel(LGBCFG)
    model.set_data(train_i)
    model.run(VAL_DATE + SUB_DATE)
    df_list.append(model.train_copy)
concat_df = pd.concat(df_list)
concat_df = concat_df.sort_index(axis="index")
# model.train = pd.concat(df_list)
# model.train = model.train.sort_index(axis="index")
# --- Post ---
# loss.csv
tmp = []
tmp2 = []
print('Calculate and save loss')

ans_val  = train[( pd.to_datetime(train['first_day_of_month']).isin(VAL_DATE))]
ans_sub  = train[( pd.to_datetime(train['first_day_of_month']).isin(SUB_DATE[1:]))]
pred_val = concat_df[( pd.to_datetime(concat_df['first_day_of_month']).isin(VAL_DATE))]
pred_sub = concat_df[( pd.to_datetime(concat_df['first_day_of_month']).isin(SUB_DATE[1:]))]

for cfip in tqdm(train['cfips'].unique()):
    tmp2.append(cfip)
    val_loss = smape(ans_val.loc[(ans_val['cfips']== cfip)]['microbusiness_density'].values,
                    pred_val.loc[(pred_val['cfips']== cfip)]['microbusiness_density'].values)
    sub_loss = smape(ans_sub.loc[(train['cfips']== cfip)]['microbusiness_density'].values,
                    pred_sub.loc[(pred_sub['cfips']== cfip)]['microbusiness_density'].values)
    tmp2.append(val_loss)
    tmp2.append(sub_loss)
    tmp.append(tmp2)
    tmp2 = []
    

loss = pd.DataFrame(tmp)
loss.columns = ['cfips', 'val_loss', 'sub_loss']
loss.to_csv(loss_filepath, index = False)



train['first_day_of_month'] = pd.to_datetime(train['first_day_of_month'])
train_period = train[(train['first_day_of_month'] >= pd.to_datetime('2021/3/1')) & (train['first_day_of_month'] <= pd.to_datetime('2022/12/1'))]

count_dic = {}
for cfip in train_period['cfips'].unique():
    previous_active = np.nan
    count = 0
    for active in train_period[train_period['cfips'] == cfip]['active'].values[10:30]:
        if previous_active==np.nan:
            previous_active = active
        else:
            if previous_active == active:
                count += 1
        previous_active = active
    count_dic[cfip] = count
    
active = train.groupby('cfips')['active'].mean().reset_index()

active['counts'] = active['cfips'].map(count_dic)

ACTIVE_TH = 2
cfips_last_val = active[active['active'] <= ACTIVE_TH]['cfips'].values
for cfip in tqdm(train['cfips'].unique()):
    tmp2.append(cfip)
    if cfip in cfips_last_val:
        mbd = train[(train['first_day_of_month'] == pd.to_datetime('2022/7/1')) & (train['cfips'] == cfip)]['microbusiness_density'].values
        val_loss = smape(ans_val.loc[(ans_val['cfips']== cfip)]['microbusiness_density'].values,
                        [mbd])
        sub_loss = smape(ans_sub.loc[(train['cfips']== cfip)]['microbusiness_density'].values,
                        [mbd for _ in range(3)])
    else:
        val_loss = smape(ans_val.loc[(ans_val['cfips']== cfip)]['microbusiness_density'].values,
                        pred_val.loc[(pred_val['cfips']== cfip)]['microbusiness_density'].values)
        sub_loss = smape(ans_sub.loc[(train['cfips']== cfip)]['microbusiness_density'].values,
                        pred_sub.loc[(pred_sub['cfips']== cfip)]['microbusiness_density'].values)
    tmp2.append(val_loss)
    tmp2.append(sub_loss)
    tmp.append(tmp2)
    tmp2 = []
    

loss = pd.DataFrame(tmp)
loss.columns = ['cfips', 'val_loss', 'sub_loss']
loss.to_csv(loss_last_filepath, index = False)  
    
    
if  LGBCFG['mode'] == 'prediction':
    VAL_DATE = ['2023/1/1']
    SUB_DATE = ['2023/2/1', '2023/3/1', '2023/4/1', '2023/5/1']
    
    # model = LGBModel(LGBCFG)
    # model.set_data(train)
    # model.run(VAL_DATE + SUB_DATE)
    df_list = []
    query_list = [low_cfips, middle_cfips, high_cfips]
    for cfips_i in query_list:
        train_i = train.loc[train["cfips"].isin(cfips_i),:]
        model = LGBModel(LGBCFG)
        model.set_data(train_i)
        model.run(VAL_DATE + SUB_DATE)
        df_list.append(model.train_copy)
    concat_df = pd.concat(df_list)
    concat_df = concat_df.sort_index(axis="index")


print('Save inference')
inference_df = pd.concat([concat_df.loc[concat_df['first_day_of_month'].isin(VAL_DATE),['microbusiness_density']].reset_index(drop = True).T] + 
                        [concat_df.loc[concat_df['first_day_of_month'].isin([pd.to_datetime(sub)]),['microbusiness_density']].reset_index(drop = True).T for sub in SUB_DATE])
# predict.csv (いいやり方ではないと思う。。。とりあえずバリデーション期間が変わってもできるように)
# inference_df.columns = model.mart_val['cfips'].values
extract_concat_df = concat_df[concat_df["first_day_of_month"] == pd.to_datetime(SUB_DATE[-1])]
inference_df.columns = extract_concat_df['cfips'].values
inference_df.to_csv(inferenced_filepath, index = False)
