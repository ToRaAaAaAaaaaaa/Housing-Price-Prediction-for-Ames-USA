import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import os

# データの読み込み
home = os.path.expanduser("~")
train_path = os.path.join(home, "OneDrive", "デスクトップ", "機械学習", "signate", "HousingPrediction", "csv", "train.csv")
test_path = os.path.join(home, "OneDrive", "デスクトップ", "機械学習", "signate", "HousingPrediction", "csv", "test.csv")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# 名義尺度の変換
unique_categories = pd.concat([train["MS SubClass"], test["MS SubClass"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["MS SubClass"] = train["MS SubClass"].map(category_map)
test["MS SubClass"] = test["MS SubClass"].map(category_map)

# 説明変数と予測変数
target = train['SalePrice'].values
features_one = train[["MS SubClass", "Lot Area", "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Gr Liv Area", "Garage Area"]].values
test_features_one = test[["MS SubClass", "Lot Area", "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Gr Liv Area", "Garage Area"]].values

# 変数の標準化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_one)
test_features_scaled = scaler.transform(test_features_one)
target_scaled = scaler.fit_transform(target.reshape(-1, 1)).ravel()

# モデルの学習と予測
svm = SVR(kernel='rbf', C=1.0, gamma='scale')
svm.fit(features_scaled, target_scaled)
my_prediction_scaled = svm.predict(test_features_scaled)

# 標準化を戻す
my_prediction = scaler.inverse_transform(my_prediction_scaled.reshape(-1, 1)).ravel()

'''
# 予測データサイズを確認
print(my_prediction.shape)
(2000,)
'''
print(my_prediction)

# csvファイルの作成
# 行単位でデータをまとめる
index = np.array(test['index']).astype(int)
data = list(zip(index, my_prediction))

# 列名無のDataFrameを作成
my_solution = pd.DataFrame(data)
my_solution.to_csv('my_tree_one.csv', index=False, header=False)

# 結果
# 27624.7714034