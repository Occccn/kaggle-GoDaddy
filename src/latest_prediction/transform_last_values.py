import pandas as pd
import os
import numpy as np

sub      = pd.read_csv('submission.csv')
sub_last = pd.read_csv('submission_last.csv')

sub['cfips']      = sub['row_id'].apply(lambda x :x.split('_')[0])
sub_last['cfips'] = sub_last['row_id'].apply(lambda x :x.split('_')[0])


# --- Set Paramters ---
data_folder = "../../data/raw"
train_filepath = os.path.join(data_folder, "train.csv")
test_filepath = os.path.join(data_folder, "test.csv")
revealed_test_filepath = os.path.join(data_folder, "revealed_test.csv")

# --- Preparation ---
# 変数の定義
train           = pd.read_csv(train_filepath)
test            = pd.read_csv(test_filepath)
revealed_test   = pd.read_csv(revealed_test_filepath)

train = pd.concat([train, revealed_test])
test  = pd.merge(test,train[['cfips', 'county', 'state']].drop_duplicates(),how = 'left')
test  = test[~test['first_day_of_month'].isin(['2022-11-01','2022-12-01'])]
train = pd.concat([train,test]).reset_index(drop = True)

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

ACTIVE_TH = 5
cfips = active[active['counts'] > ACTIVE_TH]['cfips']

print(f'last_value_model:{len(cfips)}')
print(f'ML_model:{len(train["cfips"].unique())- len(cfips)}')

for cfip in cfips:
    sub.loc[sub['cfips'] == str(cfip),'microbusiness_density'] = sub_last.loc[sub_last['cfips'] == str(cfip),'microbusiness_density']
    
sub = sub.drop(columns = 'cfips')

sub.to_csv('submission_transform_last.csv',index = False)