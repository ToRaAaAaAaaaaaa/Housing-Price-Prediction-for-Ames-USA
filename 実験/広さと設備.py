import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

'''
改善点
変数
BsmtFin SF 1（連続型）: 地下室仕上げ面積 1（平方フィート）
Bsmt Unf SF（連続型）: 地下室未仕上げ面積（平方フィート）
Total Bsmt SF（連続型）: 地下室総面積（平方フィート）
Heating QC（順序尺度）: 暖房設備の品質と状態
Central Air（名義尺度）: 中央エアコンの有無
Electrical（名義尺度）: 電気システムの種類
1st Flr SF（連続型）: 1階の床面積（平方フィート）
2nd Flr SF（連続型）: 2階の床面積（平方フィート）
Gr Liv Area（連続型）: 地上の居住面積（平方フィート）
Full Bath（離散型）: 地上のフルバスルーム数
Half Bath（離散型）: 地上のハーフバスルーム数
Bedroom AbvGr（離散型）: 地上の寝室数
Kitchen AbvGr（離散型）: 地上のキッチン数
Kitchen Qual（順序尺度）: キッチンの品質
TotRms AbvGrd（離散型）: 地上の総部屋数（バスルームを除く）
Fireplaces（離散型）: 暖炉の数
Garage Cars（離散型）: ガレージに駐車可能な車の数
Garage Area（連続型）: ガレージの面積（平方フィート）
Paved Drive（名義尺度）: 舗装された駐車場の有無
Wood Deck SF（連続型）: ウッドデッキの面積（平方フィート）
Open Porch SF（連続型）: 開放型ポーチの面積（平方フィート）
'''

'''
# データの読み込み
'''
train = pd.read_csv(r'C:\Users\itato\OneDrive\デスクトップ\機械学習\signate\【第53回_Beginner限定コンペ】アメリカの都市エイムズの住宅価格予測\csv\train.csv')
test = pd.read_csv(r'C:\Users\itato\OneDrive\デスクトップ\機械学習\signate\【第53回_Beginner限定コンペ】アメリカの都市エイムズの住宅価格予測\csv\test.csv')


'''
名義尺度の変換
'''

# Central Air
unique_categories = pd.concat([train["Central Air"], test["Central Air"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Central Air"] = train["Central Air"].map(category_map)
test["Central Air"] = test["Central Air"].map(category_map)

# Electrical
unique_categories = pd.concat([train["Electrical"], test["Electrical"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Electrical"] = train["Electrical"].map(category_map)
test["Electrical"] = test["Electrical"].map(category_map)

# Paved Drive
unique_categories = pd.concat([train["Paved Drive"], test["Paved Drive"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Paved Drive"] = train["Paved Drive"].map(category_map)
test["Paved Drive"] = test["Paved Drive"].map(category_map)

'''
順序尺度の変換
'''

# Heating QC
unique_categories = pd.concat([train["Heating QC"], test["Heating QC"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Heating QC"] = train["Heating QC"].map(category_map)
test["Heating QC"] = test["Heating QC"].map(category_map)

# Kitchen Qual
unique_categories = pd.concat([train["Kitchen Qual"], test["Kitchen Qual"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Kitchen Qual"] = train["Kitchen Qual"].map(category_map)
test["Kitchen Qual"] = test["Kitchen Qual"].map(category_map)

# 説明変数と目的変数の決定
target = train['SalePrice'].values
features_one = train[["BsmtFin SF 1", "Bsmt Unf SF", "Total Bsmt SF", "Heating QC", "Central Air", "Electrical", "1st Flr SF", "2nd Flr SF", "Gr Liv Area", "Full Bath", "Half Bath", "Bedroom AbvGr", "Kitchen AbvGr", "Kitchen Qual", "TotRms AbvGrd", "Fireplaces", "Garage Cars", "Garage Area", "Paved Drive", "Wood Deck SF", "Open Porch SF"]].values
test_features_one = test[["BsmtFin SF 1", "Bsmt Unf SF", "Total Bsmt SF", "Heating QC", "Central Air", "Electrical", "1st Flr SF", "2nd Flr SF", "Gr Liv Area", "Full Bath", "Half Bath", "Bedroom AbvGr", "Kitchen AbvGr", "Kitchen Qual", "TotRms AbvGrd", "Fireplaces", "Garage Cars", "Garage Area", "Paved Drive", "Wood Deck SF", "Open Porch SF"]].values

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