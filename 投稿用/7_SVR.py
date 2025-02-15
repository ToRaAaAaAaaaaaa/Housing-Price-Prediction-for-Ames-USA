import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import os

'''
改善点
'''

# データの読み込み
home = os.path.expanduser("~")
train_path = os.path.join(home, "OneDrive", "デスクトップ", "機械学習", "signate", "HousingPrediction", "csv", "train.csv")
test_path = os.path.join(home, "OneDrive", "デスクトップ", "機械学習", "signate", "HousingPrediction", "csv", "test.csv")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

'''
# 順序尺度の変換
'''
# Kitchen Qual
unique_categories = pd.concat([train["Kitchen Qual"], test["Kitchen Qual"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Kitchen Qual"] = train["Kitchen Qual"].map(category_map)
test["Kitchen Qual"] = test["Kitchen Qual"].map(category_map)

'''
# 名義尺度の変換
'''
# MS SubClass
unique_categories = pd.concat([train["MS SubClass"], test["MS SubClass"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["MS SubClass"] = train["MS SubClass"].map(category_map)
test["MS SubClass"] = test["MS SubClass"].map(category_map)
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

# 説明変数と予測変数
target = train['SalePrice'].values # 住宅価格
features_one = train[["Overall Qual", "Kitchen Qual", "MS SubClass", "Lot Area", "Total Bsmt SF", "1st Flr SF", "Gr Liv Area", "Garage Area", "Sale Type", "Sale Condition"]].values
test_features_one = test[["Overall Qual", "Kitchen Qual", "MS SubClass", "Lot Area", "Total Bsmt SF", "1st Flr SF", "Gr Liv Area", "Garage Area", "Sale Type", "Sale Condition"]].values
'''
変数
MS SubClass（名義尺度）: 販売対象の住宅の種類を識別
Lot Area（連続型）: 土地の面積（平方フィート）
Overall Qual（順序尺度）: 物件の全体的な素材と仕上げの品質（1～10の範囲）
Total Bsmt SF（連続型）: 地下室総面積（平方フィート）
1st Flr SF（連続型）: 1階の床面積（平方フィート）
Gr Liv Area（連続型）: 地上の居住面積（平方フィート）
Kitchen Qual（順序尺度）: キッチンの品質
Garage Area（連続型）: ガレージの面積（平方フィート）
Sale Type（名義尺度）: 売却タイプ
Sale Condition（名義尺度）: 売却条件
'''

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