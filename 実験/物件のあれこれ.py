import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

'''
改善点
変数
Neighborhood（名義尺度）: 物件が位置するエイムズ市内の地区名
Bldg Type（名義尺度）: 建物の種類
House Style（名義尺度）: 住宅のスタイル
Overall Qual（順序尺度）: 物件の全体的な素材と仕上げの品質（1～10の範囲）
Overall Cond（順序尺度）: 物件の全体的な状態（1～10の範囲）
Year Built（離散型）: 建物の建設年
Year Remod/Add（離散型）: 改築または増築が行われた年
Roof Style（名義尺度）: 屋根のスタイル
Exterior 1st（名義尺度）: 外壁材の種類（主材）
Exter Qual（順序尺度）: 外装材の品質
Foundation（名義尺度）: 基礎の種類
'''

'''
# データの読み込み
'''
train = pd.read_csv(r'C:\Users\itato\OneDrive\デスクトップ\機械学習\signate\【第53回_Beginner限定コンペ】アメリカの都市エイムズの住宅価格予測\csv\train.csv')
test = pd.read_csv(r'C:\Users\itato\OneDrive\デスクトップ\機械学習\signate\【第53回_Beginner限定コンペ】アメリカの都市エイムズの住宅価格予測\csv\test.csv')


'''
名義尺度の変換
'''
# Neighborhood
unique_categories = pd.concat([train["Neighborhood"], test["Neighborhood"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Neighborhood"] = train["Neighborhood"].map(category_map)
test["Neighborhood"] = test["Neighborhood"].map(category_map)

# Bldg Type
unique_categories = pd.concat([train["Bldg Type"], test["Bldg Type"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Bldg Type"] = train["Bldg Type"].map(category_map)
test["Bldg Type"] = test["Bldg Type"].map(category_map)

# House Style
unique_categories = pd.concat([train["House Style"], test["House Style"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["House Style"] = train["House Style"].map(category_map)
test["House Style"] = test["House Style"].map(category_map)

# Roof Style
unique_categories = pd.concat([train["Roof Style"], test["Roof Style"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Roof Style"] = train["Roof Style"].map(category_map)
test["Roof Style"] = test["Roof Style"].map(category_map)

# Exterior 1st
unique_categories = pd.concat([train["Exterior 1st"], test["Exterior 1st"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Exterior 1st"] = train["Exterior 1st"].map(category_map)
test["Exterior 1st"] = test["Exterior 1st"].map(category_map)

# Foundation
unique_categories = pd.concat([train["Foundation"], test["Foundation"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Foundation"] = train["Foundation"].map(category_map)
test["Foundation"] = test["Foundation"].map(category_map)
'''
順序尺度の変換
'''

# Overall Qual
unique_categories = pd.concat([train["Overall Qual"], test["Overall Qual"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Overall Qual"] = train["Overall Qual"].map(category_map)
test["Overall Qual"] = test["Overall Qual"].map(category_map)

# Overall Cond
unique_categories = pd.concat([train["Overall Cond"], test["Overall Cond"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Overall Cond"] = train["Overall Cond"].map(category_map)
test["Overall Cond"] = test["Overall Cond"].map(category_map)

# Exter Qual
unique_categories = pd.concat([train["Exter Qual"], test["Exter Qual"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Exter Qual"] = train["Exter Qual"].map(category_map)
test["Exter Qual"] = test["Exter Qual"].map(category_map)

# 説明変数と目的変数の決定
target = train['SalePrice'].values
features_one = train[["Neighborhood", "Bldg Type", "House Style", "Overall Qual", "Overall Cond", "Year Built", "Year Remod/Add", "Roof Style", "Exterior 1st", "Exter Qual", "Foundation"]].values
test_features_one = test[["Neighborhood", "Bldg Type", "House Style", "Overall Qual", "Overall Cond", "Year Built", "Year Remod/Add", "Roof Style", "Exterior 1st", "Exter Qual", "Foundation"]].values

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