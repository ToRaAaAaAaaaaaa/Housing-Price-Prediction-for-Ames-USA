from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os

# データの読み込み
home = os.path.expanduser("~")
train_path = os.path.join(home, "OneDrive", "デスクトップ", "機械学習", "signate", "HousingPrediction", "csv", "train.csv")
test_path = os.path.join(home, "OneDrive", "デスクトップ", "機械学習", "signate", "HousingPrediction", "csv", "test.csv")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# 順序尺度の変換
train.loc[train["Kitchen Qual"] == "po", "Kitchen Qual"] = 1
train.loc[train["Kitchen Qual"] == "Fa", "Kitchen Qual"] = 2
train.loc[train["Kitchen Qual"] == "TA", "Kitchen Qual"] = 3
train.loc[train["Kitchen Qual"] == "Gd", "Kitchen Qual"] = 4
train.loc[train["Kitchen Qual"] == "Ex", "Kitchen Qual"] = 5
test.loc[test["Kitchen Qual"] == "po", "Kitchen Qual"] = 1
test.loc[test["Kitchen Qual"] == "Fa", "Kitchen Qual"] = 2
test.loc[test["Kitchen Qual"] == "TA", "Kitchen Qual"] = 3
test.loc[test["Kitchen Qual"] == "Gd", "Kitchen Qual"] = 4
test.loc[test["Kitchen Qual"] == "Ex", "Kitchen Qual"] = 5

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
train.loc[train["Sale Type"] == "Oth", "Sale Type"] = 1
test.loc[test["Sale Type"] == "New", "Sale Type"] = 7
test.loc[test["Sale Type"] == "COD", "Sale Type"] = 6
test.loc[test["Sale Type"] == "WD ", "Sale Type"] = 5
test.loc[test["Sale Type"] == "CWD", "Sale Type"] = 4
test.loc[test["Sale Type"] == "VWD", "Sale Type"] = 3
test.loc[test["Sale Type"] == "Con", "Sale Type"] = 2
test.loc[test["Sale Type"] == "ConLw", "Sale Type"] = 1
test.loc[test["Sale Type"] == "ConLI", "Sale Type"] = 1
test.loc[test["Sale Type"] == "ConLD", "Sale Type"] = 1
test.loc[test["Sale Type"] == "Oth", "Sale Type"] = 1

train.loc[train["Sale Condition"] == "Normal", "Sale Condition"] = 6
train.loc[train["Sale Condition"] == "Abnormal", "Sale Condition"] = 5
train.loc[train["Sale Condition"] == "Partial", "Sale Condition"] = 4
train.loc[train["Sale Condition"] == "Family", "Sale Condition"] = 3
train.loc[train["Sale Condition"] == "AdjLand", "Sale Condition"] = 2
train.loc[train["Sale Condition"] == "Alloca", "Sale Condition"] = 1
test.loc[test["Sale Condition"] == "Normal", "Sale Condition"] = 6
test.loc[test["Sale Condition"] == "Abnormal", "Sale Condition"] = 5
test.loc[test["Sale Condition"] == "Partial", "Sale Condition"] = 4
test.loc[test["Sale Condition"] == "Family", "Sale Condition"] = 3
test.loc[test["Sale Condition"] == "AdjLand", "Sale Condition"] = 2
test.loc[test["Sale Condition"] == "Alloca", "Sale Condition"] = 1

# 説明変数と予測変数
target = train['SalePrice'].values
features_one = train[["Overall Qual", "Kitchen Qual", "MS SubClass", "Lot Area", "Total Bsmt SF", "1st Flr SF", "Gr Liv Area", "Garage Area", "Sale Type", "Sale Condition"]].values
test_features_one = test[["Overall Qual", "Kitchen Qual", "MS SubClass", "Lot Area", "Total Bsmt SF", "1st Flr SF", "Gr Liv Area", "Garage Area", "Sale Type", "Sale Condition"]].values


# データの準備
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_one)
target_scaled = scaler.fit_transform(target.reshape(-1, 1)).ravel()

# モデルの定義
svm = SVR(kernel='rbf', C=1.0, gamma='scale')

# k分割交差検証
scores = cross_val_score(svm, features_scaled, target_scaled, cv=5, scoring='neg_mean_squared_error')

# 平均スコアと標準偏差
print(f"平均スコア: {-np.mean(scores):.4f}")
print(f"標準偏差: {np.std(scores):.4f}")
