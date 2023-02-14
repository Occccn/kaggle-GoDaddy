import pandas as pd
import lightgbm as lgb
import numpy as np

class LGBModel:
    def __init__(self):
        self.data          = None
        self.train_x       = None
        self.train_y       = None
        self.val_x         = None
        self.val_y         = None
        self.sub_x         = None
        self.sub_y         = None
        self.predict_val   = None
        self.metric_val    = None
        self.modelinstance = None
        self.NOT_FEATURES  = ['cfips','first_day_of_month','target','shift_target','microbusiness_density', 'row_id']
        self.TARGET        = ['target']
        self.DROP_COUNTY   = []
        self.PARAMS        = {
                            'objective'        : 'regression', 
                            'metric'           : 'mae',
                            'lambda_l1'        : 2,
                            'lambda_l2'        : 5,
                            'num_leaves'       : 40,
                            'learning_rate'    : 0.005,
                            'max_depth'        : 8,
                            'min_child_samples': 150,
                            'force_col_wise'   : True,
                            'num_iterations'   : 100,
                        #     'early_stopping_round':100,
                        #     'boosting' : boosting_type,
                        #     'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 0.4),
                        #     #'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                        #     #'bagging_freq': trial.suggest_int('bagging_freq', 1, 30),
                        }
    def set_data(self, _data):
        self.train = _data
        #学習対象から一部群を除く、「48301」はターゲットがinfとなったため
        self.train = self.train[~self.train['cfips'].isin(self.DROP_COUNTY)]
        #型変換
        self.train['first_day_of_month'] = pd.to_datetime(self.train['first_day_of_month'])
    
    def feature_engioneering(self):
        feature_df = self.train[self.train['first_day_of_month'] >= pd.to_datetime('2021/2/1')][['cfips', 'first_day_of_month','row_id']].copy()
        
        ##diff(2021/1⇒2021/2)特徴量
        self.train['microbusiness_density_shift1'] = self.train.groupby('cfips')['microbusiness_density'].shift(1)
        self.train['diff']                         = self.train['microbusiness_density_shift1']- self.train['microbusiness_density']
        self.train['diff_abs']                     = self.train['diff'].abs()
        train_old                                  = self.train[self.train['first_day_of_month'] < pd.to_datetime('2021/2/1')]
        train_old_gby                              = train_old.groupby('cfips')['diff_abs'].mean().reset_index()
        train_diff                                 = self.train[self.train['first_day_of_month'] == pd.to_datetime('2021/2/1')][['cfips','diff']]
        diff_feature                               = pd.merge(train_old_gby, train_diff,on = 'cfips')
        diff_feature['diff_1_2']                   = diff_feature['diff']/diff_feature['diff_abs']
        
        
        ##lag特徴量
        lag_features     = self.train[self.train['first_day_of_month'] >= pd.to_datetime('2021/2/1')]
        lag_features_col = []
        for shift_month in range(1,len(lag_features['first_day_of_month'].unique())):
            lag_features[f'mb_dens_shift{shift_month}'] = lag_features.groupby(['cfips'])[['microbusiness_density']].shift(shift_month)
            lag_features_col.append(f'mb_dens_shift{shift_month}')

        ## 群の数
        cfips_amount                        = self.train.groupby('state')[['cfips']].nunique().reset_index().rename({"cfips":"cfips_amount"},axis = 1)
        cfips_amount                        = pd.merge(self.train[['cfips','state']],cfips_amount , on = 'state')
        cfips_amount                        = cfips_amount.groupby(['cfips'])[['cfips_amount','state']].last().reset_index()
        state_dummies                       = pd.get_dummies(cfips_amount['state'])
        cfips_amount[state_dummies.columns] = state_dummies
        cfips_amount                        = cfips_amount.drop(columns = 'state')
        
        #移動平均
        rolling_features     = self.train[self.train['first_day_of_month'] >= pd.to_datetime('2021/2/1')]
        rolling_features_col = []
        for i in range(1,2):
        # for i in range(1,6):
            DAYS_PRED = i+1
            # for size in [3, 6, 9, 12]:
            for size in [3]:
                rolling_features[f"rolling_lag_mean_t{size}_shift{DAYS_PRED}"] = rolling_features.groupby(['cfips'])['microbusiness_density'].transform(lambda x: x.shift(DAYS_PRED).rolling(size).mean())
                rolling_features_col.append(f"rolling_lag_mean_t{size}_shift{DAYS_PRED}")
                
                
        #分散
        std_features     = self.train[self.train['first_day_of_month'] >= pd.to_datetime('2021/2/1')]
        std_features_col = []
        for i in range(1,2):
        # for i in range(1,6):
            DAYS_PRED = i+1
            # for size in [3, 6, 9, 12]:
            for size in [3]:
                std_features[f"rolling_lag_std_t{size}_shift{DAYS_PRED}"] = std_features.groupby(['cfips'])['microbusiness_density'].transform(lambda x: x.shift(DAYS_PRED).rolling(size).std())
                std_features_col.append(f"rolling_lag_std_t{size}_shift{DAYS_PRED}")
                
        #ターゲット
        target                 = self.train[self.train['first_day_of_month'] >= pd.to_datetime('2021/2/1')]
        target['shift_target'] = target.groupby('cfips')['microbusiness_density'].shift(1)
        target['target']       = target['microbusiness_density']/target['shift_target'] - 1
        target.loc[target['target'] == np.inf, 'target'] = -1
        #各種特徴量をマージ
        feature_df2                          = pd.merge(feature_df,diff_feature,on = 'cfips')
        feature_df2[lag_features_col]        = lag_features[lag_features_col].values
        feature_df3                          = pd.merge(feature_df2,cfips_amount,on = 'cfips')
        feature_df3[rolling_features_col]    = rolling_features[rolling_features_col].values
        feature_df3[std_features_col]        = std_features[std_features_col].values
        feature_df3['target']                = target['target'].values
        feature_df3['shift_target']          = target['shift_target'].values
        feature_df3['microbusiness_density'] = target['microbusiness_density'].values
        self.mart                            = feature_df3[feature_df3['first_day_of_month'] >= pd.to_datetime('2021/3/1')]
        self.mart['target']                  = self.mart['target'].fillna(0)
        self.mart['shift_target']            = self.mart['shift_target'].fillna(0)
        
    def divide_data(self,date):
        self.mart_train = self.mart[self.mart['first_day_of_month'] < pd.to_datetime(date)]
        self.mart_val   = self.mart[self.mart['first_day_of_month'] == pd.to_datetime(date)]
        self.mart_sub   = self.mart[self.mart['first_day_of_month'] > pd.to_datetime(date)]
        
    def train_model(self): 
        #SMAPEでearly stopをかけるように修正
        dtrain      = lgb.Dataset(self.mart_train.drop(columns = self.NOT_FEATURES), self.mart_train['target'])
        dvalid      = lgb.Dataset(self.mart_val.drop(columns = self.NOT_FEATURES), self.mart_val['target'])
        gbm         = lgb.train(self.PARAMS, dtrain, valid_sets=[dvalid],callbacks=[lgb.log_evaluation(100)])
        self.preds  = gbm.predict(self.mart_val.drop(columns = self.NOT_FEATURES))

    def update_train_data(self):
        self.mart_val['preds']     = self.preds
        self.mart_val['preds_mbd'] = (self.mart_val['preds']+1) * self.mart_val['shift_target']
        self.train.loc[self.train['row_id'].isin(self.mart_val['row_id'].values),'microbusiness_density'] = ((self.mart_val['preds']+1) * self.mart_val['shift_target']).values
    def run(self,date_array):
        for date in date_array:
            self.feature_engioneering()
            self.divide_data(date)
            self.train_model()
            self.update_train_data()