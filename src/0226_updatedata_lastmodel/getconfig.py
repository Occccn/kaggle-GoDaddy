import pandas as pd
import json 

train = pd.read_csv('../../data/raw/train.csv')
cfips_list = train['cfips'].unique()

##群ごとに学習アルゴリズムを判定
config = {}
for cfips in cfips_list:
    # config[int(cfips)] = {"model":'LinearReression'}
    config[int(cfips)] = {"model":'LastModel'}
    # config[int(cfips)] = {"model":'LinearRegressor_active'}
    
    
with open('./config.json', mode="wt", encoding="utf-8") as f:
	json.dump(config, f, ensure_ascii=False, indent=2)