import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

# データの読み込み
home = os.path.expanduser("~")
train_path = os.path.join(home, "OneDrive", "デスクトップ", "機械学習", "signate", "HousingPrediction", "csv", "train.csv")
test_path = os.path.join(home, "OneDrive", "デスクトップ", "機械学習", "signate", "HousingPrediction", "csv", "test.csv")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# 説明変数と予測変数の決定
target = train['SalePrice'].values
features = train[["Lot Area", "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Gr Liv Area", "Garage Area"]].values

# ランダムフォレスト回帰モデルの作成
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(features, target)

# testの説明変数の値を取得
test_features = test[["Lot Area", "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Gr Liv Area", "Garage Area"]].values

# 予測
my_prediction = rf_model.predict(test_features)

# 予測データサイズを確認
print(my_prediction.shape)
print(my_prediction)

# csvファイルの作成
# 行単位でデータをまとめる
index = np.array(test['index']).astype(int)
data = list(zip(index, my_prediction))

# 列名無のDataFrameを作成
my_solution = pd.DataFrame(data)
my_solution.to_csv('my_tree_one.csv', index=False, header=False)

# 結果
# 26589.3672269
