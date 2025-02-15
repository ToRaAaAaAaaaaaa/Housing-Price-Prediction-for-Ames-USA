import pandas as pd
import numpy as np
from sklearn import tree
import os

home = os.path.expanduser("~")
train_path = os.path.join(home, "OneDrive", "デスクトップ", "機械学習", "signate", "HousingPrediction", "csv", "train.csv")
test_path = os.path.join(home, "OneDrive", "デスクトップ", "機械学習", "signate", "HousingPrediction", "csv", "test.csv")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# 説明変数と予測変数の決定
target = train['SalePrice'].values
features_one = train[["Lot Area", "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Gr Liv Area", "Garage Area"]].values

# 決定木の作成
my_tree_one = tree.DecisionTreeRegressor()
my_tree_one = my_tree_one.fit(features_one, target)
# testの説明変数の値を取得
test_features = test[["Lot Area", "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Gr Liv Area", "Garage Area"]]
my_prediction = my_tree_one.predict(test_features)

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