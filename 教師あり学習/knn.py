import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

home = os.path.expanduser("~")
train_path = os.path.join(home, "OneDrive", "デスクトップ", "機械学習", "signate", "HousingPrediction", "csv", "train.csv")
train = pd.read_csv(train_path)

# region My Custom Code Block
# 名義尺度の変換
train.loc[train["Sale Type"] == "New", "Sale Type"] = 7
train.loc[train["Sale Type"] == "COD", "Sale Type"] = 6
train.loc[train["Sale Type"] == "WD ", "Sale Type"] = 5
train.loc[train["Sale Type"] == "CWD", "Sale Type"] = 4
train.loc[train["Sale Type"] == "VWD", "Sale Type"] = 3
train.loc[train["Sale Type"] == "Con", "Sale Type"] = 2
train.loc[train["Sale Type"] == "ConLw", "Sale Type"] = 1
train.loc[train["Sale Type"] == "ConLI", "Sale Type"] = 1
train.loc[train["Sale Type"] == "ConLD", "Sale Type"] = 1

train.loc[train["Sale Condition"] == "Normal", "Sale Condition"] = 6
train.loc[train["Sale Condition"] == "Abnormal", "Sale Condition"] = 5
train.loc[train["Sale Condition"] == "Partial", "Sale Condition"] = 4
train.loc[train["Sale Condition"] == "Family", "Sale Condition"] = 3
train.loc[train["Sale Condition"] == "AdjLand", "Sale Condition"] = 2
train.loc[train["Sale Condition"] == "Alloca", "Sale Condition"] = 1

# 順序尺度の変換
train.loc[train["Kitchen Qual"] == "po", "Kitchen Qual"] = 1
train.loc[train["Kitchen Qual"] == "Fa", "Kitchen Qual"] = 2
train.loc[train["Kitchen Qual"] == "TA", "Kitchen Qual"] = 3
train.loc[train["Kitchen Qual"] == "Gd", "Kitchen Qual"] = 4
train.loc[train["Kitchen Qual"] == "Ex", "Kitchen Qual"] = 5

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
'''
# endregion

# 説明変数と予測変数
y = train['SalePrice'].values # 住宅価格
X = train[["Overall Qual", "Kitchen Qual", "MS SubClass", "Lot Area", "Total Bsmt SF", "1st Flr SF", "Gr Liv Area", "Garage Area"]].values

# データの標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# データセットを訓練セットとテストセットに分離
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

n_neighbors = range(1, 11)
for n in n_neighbors:
    # モデルの決定
    reg = KNeighborsRegressor(n_neighbors=3)

    # 訓練データと訓練ターゲットを用いて学習
    reg.fit(X_train, y_train)

    # print('Test set scores:\n{}'.format(reg.predict(X_test)))
    print('Test set R^2: {:.2f}'.format(reg.score(X_test, y_test)))

# k-最近傍法アルゴリズムは処理速度が遅く、多数の特徴力を扱うことができない