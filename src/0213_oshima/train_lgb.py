import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os

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


# --- FeatureEngineering ---

#型変換
train['first_day_of_month'] = pd.to_datetime(train['first_day_of_month'])

#2021/2以降のデータを抽出
feature_df = train[train['first_day_of_month'] >= pd.to_datetime('2021/2/1')][['cfips', 'first_day_of_month']].copy()

##diff(2021/1⇒2021/2)特徴量
train['microbusiness_density_shift1'] = train.groupby('cfips')['microbusiness_density'].shift(1)
train['diff'] = train['microbusiness_density_shift1']- train['microbusiness_density']
train['diff_abs'] = train['diff'].abs()
train_old = train[train['first_day_of_month'] < pd.to_datetime('2021/2/1')]
train_old_gby = train_old.groupby('cfips')['diff_abs'].mean().reset_index()
train_diff = train[train['first_day_of_month'] == pd.to_datetime('2021/2/1')][['cfips','diff']]
diff_feature = pd.merge(train_old_gby, train_diff,on = 'cfips')
diff_feature['diff_1_2'] = diff_feature['diff']/diff_feature['diff_abs']

##lag特徴量
lag_features = train[train['first_day_of_month'] >= pd.to_datetime('2021/2/1')]
lag_features_col = []
for shift_month in range(1,len(lag_features['first_day_of_month'].unique())):
    lag_features[f'mb_dens_shift{shift_month}'] = lag_features.groupby(['cfips'])[['microbusiness_density']].shift(shift_month)
    lag_features_col.append(f'mb_dens_shift{shift_month}')
    
## 群の数
cfips_amount = train.groupby('state')[['cfips']].nunique().reset_index().rename({"cfips":"cfips_amount"},axis = 1)
cfips_amount = pd.merge(train[['cfips','state']],cfips_amount , on = 'state')
cfips_amount = cfips_amount.groupby(['cfips'])[['cfips_amount','state']].last().reset_index()
state_dummies = pd.get_dummies(cfips_amount['state'])
cfips_amount[state_dummies.columns] =state_dummies
cfips_amount = cfips_amount.drop(columns = 'state')

#移動平均
rolling_features = train[train['first_day_of_month'] >= pd.to_datetime('2021/2/1')]
rolling_features_col = []
for i in range(1,6):
    DAYS_PRED = i+1
    for size in [3, 6, 9, 12]:
        rolling_features[f"rolling_lag_mean_t{size}_shift{DAYS_PRED}"] = rolling_features.groupby(['cfips'])['microbusiness_density'].transform(lambda x: x.shift(DAYS_PRED).rolling(size).mean())
        rolling_features_col.append(f"rolling_lag_mean_t{size}_shift{DAYS_PRED}")
        
#分散
std_features = train[train['first_day_of_month'] >= pd.to_datetime('2021/2/1')]
std_features_col = []
for i in range(1,6):
    DAYS_PRED = i+1
    for size in [3, 6, 9, 12]:
        std_features[f"rolling_lag_std_t{size}_shift{DAYS_PRED}"] = std_features.groupby(['cfips'])['microbusiness_density'].transform(lambda x: x.shift(DAYS_PRED).rolling(size).std())
        std_features_col.append(f"rolling_lag_std_t{size}_shift{DAYS_PRED}")
        
#ターゲット
target = train[train['first_day_of_month'] >= pd.to_datetime('2021/2/1')]
target['shift_target'] = target.groupby('cfips')['microbusiness_density'].shift(1)
target['target'] = target['microbusiness_density']/target['shift_target'] - 1

#各種特徴量をマージ
feature_df2 = pd.merge(feature_df,diff_feature,on = 'cfips')
feature_df2[lag_features_col] = lag_features[lag_features_col].values
feature_df3 = pd.merge(feature_df2,cfips_amount,on = 'cfips')
feature_df3[rolling_features_col] = rolling_features[rolling_features_col].values
feature_df3[std_features_col] = std_features[std_features_col].values
feature_df3['target'] = target['target'].values
feature_df3['shift_target'] = target['shift_target'].values
feature_df3['microbusiness_density'] = target['microbusiness_density'].values
mart = feature_df3[feature_df3['first_day_of_month'] >= pd.to_datetime('2021/3/1')]
mart['target'] = mart['target'].fillna(0)
mart['shift_target'] = mart['shift_target'].fillna(0)

#Train/Validation/Submissionにデータ分ける
mart_train = mart[mart['first_day_of_month'] <= pd.to_datetime('2022/6/1')]
mart_val = mart[mart['first_day_of_month'] == pd.to_datetime('2022/7/1')]
mart_sub = mart[mart['first_day_of_month'] > pd.to_datetime('2022/7/1')]


#学習⇒validation期間の推論
mart_train = mart_train[~mart_train['cfips'].isin([48301])]
mart_val = mart_val[~mart_val['cfips'].isin([48301])]

dtrain = lgb.Dataset(mart_train.drop(columns = ['cfips','first_day_of_month','target','shift_target','microbusiness_density']), mart_train['target'])
dvalid = lgb.Dataset(mart_val.drop(columns = ['cfips','first_day_of_month','target','shift_target','microbusiness_density']), mart_val['target'])
param = {
    'objective': 'regression', 
    'metric': 'mae',
    'lambda_l1': 2,
    'lambda_l2': 5,
    'num_leaves': 40,
    'learning_rate': 0.005,
#     'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 0.4),
#     #'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
#     #'bagging_freq': trial.suggest_int('bagging_freq', 1, 30),
    'max_depth': 8,
    'min_child_samples': 150,
    'force_col_wise': True,
    'num_iterations' : 1000,
#     'early_stopping_round':100,
#     'boosting' : boosting_type
}

gbm = lgb.train(param, dtrain, valid_sets=[dvalid],callbacks=[lgb.log_evaluation(100)])
preds = gbm.predict(mart_val.drop(columns = ['cfips','first_day_of_month','target','shift_target','microbusiness_density']))

mart_val['preds'] = preds
mart_val['preds_mbd'] = (mart_val['preds']+1) * mart_val['shift_target']


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