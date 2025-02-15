import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

'''
# データの読み込み
'''
train = pd.read_csv(r'C:\Users\itato\OneDrive\デスクトップ\機械学習\signate\【第53回_Beginner限定コンペ】アメリカの都市エイムズの住宅価格予測\csv\train.csv')
test = pd.read_csv(r'C:\Users\itato\OneDrive\デスクトップ\機械学習\signate\【第53回_Beginner限定コンペ】アメリカの都市エイムズの住宅価格予測\csv\test.csv')

# region My Custom Code Block

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

# Lot Shape
unique_categories = pd.concat([train["Lot Shape"], test["Lot Shape"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["Lot Shape"] = train["Lot Shape"].map(category_map)
test["Lot Shape"] = test["Lot Shape"].map(category_map)

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
# endregion

# 説明変数と目的変数の決定
target = train['SalePrice'].values
features_one = train[["Lot Area", "Lot Shape", "Land Contour", "Lot Config", "Neighborhood", "Bldg Type", "House Style", "Overall Qual", "Overall Cond", "Year Built", "Year Remod/Add", "Roof Style", "Exterior 1st", "Exter Qual", "Foundation", "BsmtFin SF 1", "Bsmt Unf SF", "Total Bsmt SF", "Heating QC", "Central Air", "Electrical", "1st Flr SF", "2nd Flr SF", "Gr Liv Area", "Full Bath", "Half Bath", "Bedroom AbvGr", "Kitchen AbvGr", "Kitchen Qual", "TotRms AbvGrd", "Fireplaces", "Garage Cars", "Garage Area", "Paved Drive", "Wood Deck SF", "Open Porch SF", "Mo Sold", "Yr Sold", "Sale Type", "Sale Condition"]].values

# 変数の標準化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_one)


# svrモデル
svr = SVR(kernel='rbf', C=100, gamma=0.1)

# 非線形回帰モデル
scores = cross_val_score(svr, features_scaled, target, cv=10, scoring='r2')
print('score : {}'.format(scores))