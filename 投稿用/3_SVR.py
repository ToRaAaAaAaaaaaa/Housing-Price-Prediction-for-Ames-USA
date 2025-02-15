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

# 説明変数と予測変数の決定
target = train['SalePrice'].values
features = train[["Lot Area", "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Gr Liv Area", "Garage Area"]].values

# testの説明変数の値を取得
test_features = test[["Lot Area", "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Gr Liv Area", "Garage Area"]].values

# データの標準化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
test_features_scaled = scaler.transform(test_features)

# モデルの定義（RBF カーネル）
svm = SVR(kernel='rbf', C=1.0, gamma='scale')

# モデルの学習
svm.fit(features_scaled, target)

# 予測
my_prediction = svm.predict(test_features_scaled)

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
# 39000