import numpy as np
import pandas as pd
import yaml 
import os 
from tqdm import tqdm

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
inferenced_val_filepath = os.path.join(save_dirpath, "inferenced_val.csv")
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
train = pd.concat([train,test]).reset_index(drop = True)





with open('ensemble_eval.yml', 'r') as yml:
    CFG = yaml.load(yml, Loader=yaml.SafeLoader)


names  = CFG['model_dir']
rate   = CFG['rate']


for i in range(len(names)):
    name = names[i]
    if i == 0:
        inf = pd.read_csv(f'../{name}/train/inferenced_val.csv')
        inf *= rate[i]
    else:
        inf_tmp = pd.read_csv(f'../{name}/train/inferenced_val.csv')
        inf += inf_tmp*rate[i]
inf /= np.sum(rate)



##smapeの定義
def smape(y_pred, y_true):
    # CONVERT TO NUMPY
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # WHEN BOTH EQUAL ZERO, METRIC IS ZERO
    both = np.abs(y_true) + np.abs(y_pred)
    idx = np.where(both==0)[0]
    y_true[idx]=1; y_pred[idx]=1
    
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
##
SUB_DATE = ['2022/9/1', '2022/10/1', '2022/11/1', '2022/12/1']

ans_sub  = train[( pd.to_datetime(train['first_day_of_month']).isin(SUB_DATE[1:]))]


tmp = []
tmp2 = []
for cfip in tqdm(train['cfips'].unique()):
    tmp2.append(cfip)
    sub_loss = smape(ans_sub.loc[(train['cfips']== cfip)]['microbusiness_density'].values,
                    inf[str(cfip)][2:].values)
    tmp2.append(sub_loss)
    tmp.append(tmp2)
    tmp2 = []
    
loss = pd.DataFrame(tmp)
loss.columns = ['cfips', 'sub_loss']
loss.to_csv("ensemble_loss.csv", index = False)  
