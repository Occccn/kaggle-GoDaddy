import os

import pandas as pd
import lightgbm as lgb
import numpy as np
import optuna
from recursive_model import RecursiveModel



class LGBModel(RecursiveModel):

    def train_model(self): 
        #SMAPEでearly stopをかけるように修正
        dtrain      = lgb.Dataset(self.mart_train.drop(columns = self.NOT_FEATURES), self.mart_train[self.TARGET])
        dvalid      = lgb.Dataset(self.mart_val.drop(columns = self.NOT_FEATURES), self.mart_val[self.TARGET])
        gbm         = lgb.train(self.PARAMS, dtrain, valid_sets=[dvalid],callbacks=[lgb.log_evaluation(self.LOG_EVAL)])
        self.preds  = gbm.predict(self.mart_val.drop(columns = self.NOT_FEATURES))

    
    def tuning(self, num_trial):
        def calc_loss(_val_date, _sub_date):
            def smape(y_pred, y_true):
                # CONVERT TO NUMPY
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                
                # WHEN BOTH EQUAL ZERO, METRIC IS ZERO
                both = np.abs(y_true) + np.abs(y_pred)
                idx = np.where(both==0)[0]
                y_true[idx]=1; y_pred[idx]=1
                
                return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

            print('Calculate and save loss')
            loss_info_list = []
            # tmp = []
            # tmp2 = []
            
            ans_val  = self.org_train[( pd.to_datetime(self.org_train['first_day_of_month']).isin(_val_date))]
            ans_sub  = self.org_train[( pd.to_datetime(self.org_train['first_day_of_month']).isin(_sub_date))]
            pred_val = self.train[( pd.to_datetime(self.train['first_day_of_month']).isin(_val_date))]
            pred_sub = self.train[( pd.to_datetime(self.train['first_day_of_month']).isin(_sub_date))]

            for cfip in self.org_train['cfips'].unique():
                val_loss = smape(ans_val.loc[(ans_val['cfips']== cfip)]['microbusiness_density'].values,
                                pred_val.loc[(pred_val['cfips']== cfip)]['microbusiness_density'].values)
                sub_loss = smape(ans_sub.loc[(self.org_train['cfips']== cfip)]['microbusiness_density'].values,
                                pred_sub.loc[(pred_sub['cfips']== cfip)]['microbusiness_density'].values)

                loss_info_list.append([cfip, val_loss, sub_loss])
                
            return loss_info_list
            
        def objective(trial):
            optimized_filepath = "./result_optuna.csv"
            param = {
            'objective': 'regression',
            'metric': 'mae',
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 0.1),
            'max_depth': trial.suggest_int("max_depth", 3, 8),
            'min_child_samples': trial.suggest_int("min_child_samples", 30, 200),
            'force_col_wise': True,
            'num_iterations': trial.suggest_int('num_iterations', 100, 600),
            "verbose":-1
            }
            
            # Set Params
            self.PARAMS = param
            # train & predict
            ## Period1
            val_date1 = ['2022/8/1']
            sub_date1 = ['2022/9/1', '2022/10/1', '2022/11/1', '2022/12/1']
            self.set_data(self.org_train)
            self.run(val_date1 + sub_date1)
            loss_info_list1 = calc_loss(val_date1, sub_date1[1:])
            ## Period2
            val_date2 = ["2022/3/1"]
            sub_date2 = ["2022/4/1", "2022/5/1", "2022/6/1", '2022/7/1']
            self.set_data(self.org_train)
            self.run(val_date2 + sub_date2)
            loss_info_list2 = calc_loss(val_date2, sub_date2[1:])
            # metric
            val_loss1_list = []
            sub_loss1_list = []
            val_loss2_list = []
            sub_loss2_list = []
            val_loss_mean_list = []
            sub_loss_mean_list = []
            for i in range(len(loss_info_list1)):
                val_loss1 = loss_info_list1[i][1]
                val_loss2 = loss_info_list2[i][1]
                sub_loss1 = loss_info_list1[i][2]
                sub_loss2 = loss_info_list2[i][2]
                val_loss_mean = np.mean([val_loss1, val_loss2])
                sub_loss_mean = np.mean([sub_loss1, sub_loss2])
                val_loss1_list.append(val_loss1)
                val_loss2_list.append(val_loss2)
                sub_loss1_list.append(sub_loss1)
                sub_loss2_list.append(sub_loss2)
                val_loss_mean_list.append(val_loss_mean)
                sub_loss_mean_list.append(sub_loss_mean)
            mean_val_loss1 = sum(val_loss1_list) / len(val_loss1_list)
            mean_val_loss2 = sum(val_loss2_list) / len(val_loss2_list)
            mean_val_mean  = sum(val_loss_mean_list) / len(val_loss_mean_list)
            mean_sub_loss1 = sum(sub_loss1_list) / len(sub_loss1_list)
            mean_sub_loss2 = sum(sub_loss2_list) / len(sub_loss2_list)
            mean_sub_mean  = sum(sub_loss_mean_list) / len(sub_loss_mean_list)
            
            # Post
            if not os.path.exists(optimized_filepath):
                with open(optimized_filepath, mode="w") as f:
                    f.write("mean_val_loss1, mean_sub_loss1, mean_val_loss2, mean_sub_loss_2\n")
                    f.write(f"{mean_val_loss1}, {mean_sub_loss1}, {mean_val_loss2}, {mean_sub_loss2}\n")
            else:
                with open(optimized_filepath, mode="a") as f:
                    f.write(f"{mean_val_loss1}, {mean_sub_loss1}, {mean_val_loss2}, {mean_sub_loss2}\n")        
            return mean_sub_mean
        
        # Optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=num_trial)
        
        # Result
        return study.best_params, study.get_trials()
        