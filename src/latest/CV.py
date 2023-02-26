class CV:
    def __init__(self, _cv_type, _train):
        self.cv_type = _cv_type
        self.train = _train
        
    def divide_data(self):
        # 学習用、評価用（public）用、提出（private)用でインデックスを分ける
        if (self.cv_type == None):
            # Ref
            # https://www.kaggle.com/code/cdeotte/seasonal-model-with-validation-lb-1-091
            train_month = self.train["first_day_of_month"].values[:-4]
            val_month = self.train["first_day_of_month"].values[-4]
            submit_month = self.train["first_day_of_month"].values[-3:]
            
            train_index = self.train.query("first_day_of_month in @train_month").index
            val_index = self.train.query("first_day_of_month in @val_month").index
            submit_index = self.train.query("first_day_of_month in @submit_month").index
            
            return train_index, val_index, submit_index
            # プライベート　後ろ3ヶ月分[-3:]
            # パブリック　後ろ4ヶ月目（１ヶ月）[-4]
            # train は後ろ4 ヶ月以外[:-4]
            # index = ///
            # a = train.query("index in @index")

            # train_month = self.train["first_day_of_month"].values[-39:-1*validation_month]
            # val_month = self.train["first_day_of_month"].values[-1*validation_month:]
            
            # train_size = len(train_month)
            # val_size = len(val_month)
            # train_x = np.arrange(train_size).reshape((-1,1))
            # val_x = np.arrange(val_size).reshape((-1, 1))
            # train_y = self.train["microbusiness_density"].values
            # val_y = self.z

            # 返り値は、各データ（train, public, private）相当のインデックスを返す
        elif (self.cv_type == "from_202102"):
            """
            2021/02以降のデータで学習するようにデータを分割した分岐
            """
            train_month = self.train["first_day_of_month"].values[18:-4]  # 2021/2は、index=18なので、18以降を取得
            val_month = self.train["first_day_of_month"].values[-4]
            submit_month = self.train["first_day_of_month"].values[-3:]
            
            train_index = self.train.query("first_day_of_month in @train_month").index
            val_index = self.train.query("first_day_of_month in @val_month").index
            submit_index = self.train.query("first_day_of_month in @submit_month").index
            
            return train_index, val_index, submit_index
        
        elif (self.cv_type == "submit_full"):
            """
            全データを学習データとして用いて、予測
            """
            # !! データの変更があった際は、参照インデックス要見直し
            train_month = self.train["first_day_of_month"].values[:-8]  # 2022-10-01が-8までなので(!!データの更新があった際は要確認!!)
            val_month = self.train["first_day_of_month"].values[-8] # 2022-11-01想定（暫定）
            submit_month = self.train["first_day_of_month"].values[-7:-4] # 2022-12-01 - 2022-02-01 想定（暫定）
            
            train_index = self.train.query("first_day_of_month in @train_month").index
            val_index = self.train.query("first_day_of_month in @val_month").index
            submit_index = self.train.query("first_day_of_month in @submit_month").index
            
            return train_index, val_index, submit_index
        
        elif (self.cv_type == "submit_full_ver2"):
            """
            更新データを含んで、
            全データを学習データとして用いて、予測
            予測対象期間：
            val: 1月
            sub: 2,3,4月
            """
            # !! データの変更があった際は、参照インデックス要見直し
            train_month = self.train["first_day_of_month"].values[:-6]  # 2022-12-01が-6までなので
            val_month = self.train["first_day_of_month"].values[-6] # 2023-01-01想定
            submit_month = self.train["first_day_of_month"].values[-5:-2] # 2023-02-01 - 2022-04-01 想定
            
            train_index = self.train.query("first_day_of_month in @train_month").index
            val_index = self.train.query("first_day_of_month in @val_month").index
            submit_index = self.train.query("first_day_of_month in @submit_month").index
            
            return train_index, val_index, submit_index
        
        elif (self.cv_type == "submit_from202102_ver2"):
            """
            更新データを含んで、
            全データを学習データとして用いて、予測
            予測対象期間：
            val: 1月
            sub: 2,3,4月
            """
            # !! データの変更があった際は、参照インデックス要見直し
            train_month = self.train["first_day_of_month"].values[18:-6]  # 2021/02 - 2022-12-01
            val_month = self.train["first_day_of_month"].values[-6] # 2023-01-01想定
            submit_month = self.train["first_day_of_month"].values[-5:-2] # 2023-02-01 - 2022-04-01 想定
            
            train_index = self.train.query("first_day_of_month in @train_month").index
            val_index = self.train.query("first_day_of_month in @val_month").index
            submit_index = self.train.query("first_day_of_month in @submit_month").index
            
            return train_index, val_index, submit_index
        