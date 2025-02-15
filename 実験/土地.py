import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

'''
改善点
変数
Lot Area（連続型）: 土地の面積（平方フィート）
Lot Shape（順序尺度）: 土地の形状
Land Contour（名義尺度）: 土地の平坦さ
Lot Config（名義尺度）: 土地の構成
'''

'''
# データの読み込み
'''
train = pd.read_csv(r'C:\Users\itato\OneDrive\デスクトップ\機械学習\signate\【第53回_Beginner限定コンペ】アメリカの都市エイムズの住宅価格予測\csv\train.csv')
test = pd.read_csv(r'C:\Users\itato\OneDrive\デスクトップ\機械学習\signate\【第53回_Beginner限定コンペ】アメリカの都市エイムズの住宅価格予測\csv\test.csv')


'''
名義尺度の変換
'''
# Land Contour
unique_categories = pd.concat([train["Land Contour"], test["Land Contour"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Land Contour"] = train["Land Contour"].map(category_map)
test["Land Contour"] = test["Land Contour"].map(category_map)

# Lot Config
unique_categories = pd.concat([train["Lot Config"], test["Lot Config"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Lot Config"] = train["Lot Config"].map(category_map)
test["Lot Config"] = test["Lot Config"].map(category_map)

'''
順序尺度の変換
'''
# Lot Shape
unique_categories = pd.concat([train["Lot Shape"], test["Lot Shape"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Lot Shape"] = train["Lot Shape"].map(category_map)
test["Lot Shape"] = test["Lot Shape"].map(category_map)

# 説明変数と目的変数の決定
target = train['SalePrice'].values
features_one = train[["Lot Area", "Lot Shape", "Land Contour", "Lot Config"]].values
test_features_one = test[["Lot Area", "Lot Shape", "Land Contour", "Lot Config"]].values

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
print(my_prediction)

# csvファイルの作成
# 行単位でデータをまとめる
index = np.array(test['index']).astype(int)
data = list(zip(index, my_prediction))

# 列名無のDataFrameを作成
my_solution = pd.DataFrame(data)
my_solution.to_csv('my_tree_one.csv', index=False, header=False)

# 結果 27,376.4038000