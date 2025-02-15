import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

'''
改善点
変数
Mo Sold（離散型）: 物件が売却された月
Yr Sold（離散型）: 物件が売却された年
Sale Type（名義尺度）: 売却タイプ
Sale Condition（名義尺度）: 売却条件
'''

'''
# データの読み込み
'''
train = pd.read_csv(r'C:\Users\itato\OneDrive\デスクトップ\機械学習\signate\【第53回_Beginner限定コンペ】アメリカの都市エイムズの住宅価格予測\csv\train.csv')
test = pd.read_csv(r'C:\Users\itato\OneDrive\デスクトップ\機械学習\signate\【第53回_Beginner限定コンペ】アメリカの都市エイムズの住宅価格予測\csv\test.csv')


'''
名義尺度の変換
'''
# Sale Type
unique_categories = pd.concat([train["Sale Type"], test["Sale Type"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Sale Type"] = train["Sale Type"].map(category_map)
test["Sale Type"] = test["Sale Type"].map(category_map)

# Sale Condition
unique_categories = pd.concat([train["Sale Condition"], test["Sale Condition"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Sale Condition"] = train["Sale Condition"].map(category_map)
test["Sale Condition"] = test["Sale Condition"].map(category_map)

'''
順序尺度の変換
'''

# 説明変数と目的変数の決定
target = train['SalePrice'].values
features_one = train[["Mo Sold", "Yr Sold", "Sale Type", "Sale Condition"]].values
test_features_one = test[["Mo Sold", "Yr Sold", "Sale Type", "Sale Condition"]].values

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