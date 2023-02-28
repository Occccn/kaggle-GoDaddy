import yaml 
import pandas as pd
import numpy as np


with open('config.yml', 'r') as yml:
    CFG = yaml.load(yml, Loader=yaml.SafeLoader)
    
sub = pd.read_csv(f'../../data/raw/sample_submission.csv')

names  = CFG['model_dir']
loss_merge = pd.DataFrame()
infs   = {}
losses = {}


for name in names:
    inf        = pd.read_csv(f'../{name}/train/inferenced.csv')
    infs[name] = inf
    
    loss = pd.read_csv(f'../{name}/train/loss.csv')
    loss.columns = ['cfips',f'{name}_val_loss',f'{name}_sub_loss']
    losses[name] = loss
    
    
min_loss_model = pd.DataFrame()
min_loss_model['cfips'] = loss['cfips']


for name in names:
    min_loss_model[f'{name}_sub_loss'] = losses[name][f'{name}_sub_loss']
    if len(min_loss_model.columns) == 2:
        min_loss_model['min_loss']         = losses[name][f'{name}_sub_loss']
        min_loss_model['min_Loss_model']   = name
    else:
        update_min_loss_idx = min_loss_model['min_loss']          > min_loss_model[f'{name}_sub_loss'] 
        min_loss_model.loc[update_min_loss_idx, 'min_loss']       = min_loss_model.loc[update_min_loss_idx, f'{name}_sub_loss']
        min_loss_model.loc[update_min_loss_idx, 'min_Loss_model'] = name
        
        
sub['cfips'] = sub['row_id'].apply(lambda x : x.split('_')[0])
sub['date'] = sub['row_id'].apply(lambda x : x.split('_')[1])
cfiplist = sub['cfips'].unique()


pred_list = []
for cfip in cfiplist:
    model = min_loss_model[min_loss_model['cfips'] == int(cfip)]['min_Loss_model'].values[0]
    pred_list = np.hstack([pred_list, infs[model].loc[:,str(cfip)].values])    
    
pred_df = pd.DataFrame()
tmp = []
sort = []
for cfip in min_loss_model['cfips']:
    tmp.append(cfip);tmp.append(cfip);tmp.append(cfip);tmp.append(cfip);tmp.append(cfip)
    sort.append('2023-01-01');sort.append('2023-02-01');sort.append('2023-03-01');sort.append('2023-04-01');sort.append('2023-05-01')
pred_df['cfips'] =tmp
pred_df['date'] = sort
pred_df['pred'] = pred_list
pred_df.columns = ['cfips','date','pred_']
pred_df['row_id'] = pred_df.apply(lambda x :str(x['cfips'] )+ "_" + x['date'], axis = 1)
sub = pd.merge(sub,pred_df[['row_id','pred_']],how = 'left',on = 'row_id')
sub['pred_']= sub['pred_'].fillna(3.817671)
sub = sub[['row_id','pred_']]
sub.columns = ['row_id','microbusiness_density']



COLS = ['GEO_ID','NAME','S0101_C01_026E']
# df2020 = pd.read_csv('../../data/raw/ACSST5Y2020.S0101-Data.csv',usecols=COLS)
df2020 = pd.read_csv('../../data/raw/ACSST5Y2020.S0101-Data.csv',usecols=COLS, low_memory=False)
df2020 = df2020.iloc[1:]
df2020['S0101_C01_026E'] = df2020['S0101_C01_026E'].astype('int')

# df2021 = pd.read_csv('../../data/raw/ACSST5Y2021.S0101-Data.csv',usecols=COLS)
df2021 = pd.read_csv('../../data/raw/ACSST5Y2021.S0101-Data.csv',usecols=COLS, low_memory=False)
df2021 = df2021.iloc[1:]
df2021['S0101_C01_026E'] = df2021['S0101_C01_026E'].astype('int')

df2020['cfips'] = df2020['GEO_ID'].apply(lambda x: int(x.split('US')[-1]) )
adult2020 = df2020.set_index('cfips').S0101_C01_026E.to_dict()

df2021['cfips'] = df2021['GEO_ID'].apply(lambda x: int(x.split('US')[-1]) )
adult2021 = df2021.set_index('cfips').S0101_C01_026E.to_dict()

sub['cfips'] = sub['row_id'].apply(lambda x: int(x.split('_')[0]))
sub['adult2020'] = sub['cfips'].map(adult2020)
sub['adult2021'] = sub['cfips'].map(adult2021)
sub.head()

sub['microbusiness_density'] = sub['microbusiness_density'] * sub['adult2020'] / sub['adult2021']
sub= sub.drop(['adult2020','adult2021','cfips'],axis=1)
sub.to_csv('submission.csv',index = False)
