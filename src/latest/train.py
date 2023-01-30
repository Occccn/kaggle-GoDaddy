import pandas as pd
import json
	
train           = pd.read_csv('../../data/raw/train.csv')
test            = pd.read_csv('../../data/raw/test.csv')
submission      = pd.read_csv('../../data/raw/sample_submission.csv')
census_starter  = pd.read_csv('../../data/raw/census_starter.csv')


with open('./config.json', mode="rt", encoding="utf-8") as f:
	config = json.load(f)

#イメージ
def judge_model(model_type):
    pass

#群ごとに予測を実施
for cfips_config in config:
    model_type = cfips_config['model']
    
    judge_model(model_type)
        