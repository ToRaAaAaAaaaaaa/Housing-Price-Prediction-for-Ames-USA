import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error


home = os.path.expanduser("~")
train_path = os.path.join(home, "OneDrive", "デスクトップ", "機械学習", "signate", "HousingPrediction", "csv", "train.csv")
test_path = os.path.join(home, "OneDrive", "デスクトップ", "機械学習", "signate", "HousingPrediction", "csv", "test.csv")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


# region My Custom Code Block

'''
ダミー変数の作成
'''
def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name, drop_first=True)  # drop_first=Falseですべてのカテゴリーを保持
    dummies = dummies.applymap(lambda x: 1 if x > 0 else 0)
    df = pd.concat([df, dummies], axis=1)
    return df

'''
名義尺度の変換
'''

train = create_dummies(train, 'MS SubClass')
train = create_dummies(train, 'MS Zoning')
train = create_dummies(train, 'Land Contour')
train = create_dummies(train, 'Lot Config')
train = create_dummies(train, 'Neighborhood')
train = create_dummies(train, 'Bldg Type')
train = create_dummies(train, 'House Style')
train = create_dummies(train, 'Roof Style')
train = create_dummies(train, 'Exterior 1st')
train = create_dummies(train, 'Foundation')
train = create_dummies(train, 'Central Air')
train = create_dummies(train, 'Electrical')
train = create_dummies(train, 'Paved Drive')
train = create_dummies(train, 'Sale Type')
train = create_dummies(train, 'Sale Condition')

test = create_dummies(test, 'MS SubClass')
test = create_dummies(test, 'MS Zoning')
test = create_dummies(test, 'Land Contour')
test = create_dummies(test, 'Lot Config')
test = create_dummies(test, 'Neighborhood')
test = create_dummies(test, 'Bldg Type')
test = create_dummies(test, 'House Style')
test = create_dummies(test, 'Roof Style')
test = create_dummies(test, 'Exterior 1st')
test = create_dummies(test, 'Foundation')
test = create_dummies(test, 'Central Air')
test = create_dummies(test, 'Electrical')
test = create_dummies(test, 'Paved Drive')
test = create_dummies(test, 'Sale Type')
test = create_dummies(test, 'Sale Condition')

'''
順序尺度の変換
'''

# Lot Shape
train.loc[train["Lot Shape"] == "Reg", "Lot Shape"] = 4
train.loc[train["Lot Shape"] == "IR1", "Lot Shape"] = 3
train.loc[train["Lot Shape"] == "IR2", "Lot Shape"] = 2
train.loc[train["Lot Shape"] == "IR3", "Lot Shape"] = 1
test.loc[test["Lot Shape"] == "Reg", "Lot Shape"] = 4
test.loc[test["Lot Shape"] == "IR1", "Lot Shape"] = 3
test.loc[test["Lot Shape"] == "IR2", "Lot Shape"] = 2
test.loc[test["Lot Shape"] == "IR3", "Lot Shape"] = 1

# Exter Qual
train.loc[train["Exter Qual"] == "Ex", "Exter Qual"] = 5
train.loc[train["Exter Qual"] == "Gd", "Exter Qual"] = 4
train.loc[train["Exter Qual"] == "TA", "Exter Qual"] = 3
train.loc[train["Exter Qual"] == "Fa", "Exter Qual"] = 2
train.loc[train["Exter Qual"] == "Po", "Exter Qual"] = 1

test.loc[test["Exter Qual"] == "Ex", "Exter Qual"] = 5
test.loc[test["Exter Qual"] == "Gd", "Exter Qual"] = 4
test.loc[test["Exter Qual"] == "TA", "Exter Qual"] = 3
test.loc[test["Exter Qual"] == "Fa", "Exter Qual"] = 2
test.loc[test["Exter Qual"] == "Po", "Exter Qual"] = 1

# Heating QC
train.loc[train["Heating QC"] == "Ex", "Heating QC"] = 5
train.loc[train["Heating QC"] == "Gd", "Heating QC"] = 4
train.loc[train["Heating QC"] == "TA", "Heating QC"] = 3
train.loc[train["Heating QC"] == "Fa", "Heating QC"] = 2
train.loc[train["Heating QC"] == "Po", "Heating QC"] = 1

test.loc[test["Heating QC"] == "Ex", "Heating QC"] = 5
test.loc[test["Heating QC"] == "Gd", "Heating QC"] = 4
test.loc[test["Heating QC"] == "TA", "Heating QC"] = 3
test.loc[test["Heating QC"] == "Fa", "Heating QC"] = 2
test.loc[test["Heating QC"] == "Po", "Heating QC"] = 1

# Kitchen Qual
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

# 新しいファイルパスを指定
new_train_path = os.path.join(home, "OneDrive", "デスクトップ", "機械学習", "signate", "HousingPrediction", "csv", "new_train.csv")
new_test_path = os.path.join(home, "OneDrive", "デスクトップ", "機械学習", "signate", "HousingPrediction", "csv", "new_test.csv")

# データフレームをCSVとして保存
train.to_csv(new_train_path, index=False, encoding='utf-8')
train.to_csv(new_test_path, index=False, encoding='utf-8')
# endregion

# 特徴量の決定
new_train = pd.read_csv(new_train_path)
new_test = pd.read_csv(new_test_path)
X = new_train[[
 #名義尺度（ダミー変数で変更を施した）
 'Bldg Type_2fmCon',
 'Exterior 1st_VinylSd',
 'Electrical_FuseF',
#  連続型
 'Lot Area',  
 'Total Bsmt SF', 
 '1st Flr SF', 
 '2nd Flr SF', 
 'Gr Liv Area', 
 'Open Porch SF', 
#  順序型
 'Lot Shape', 
 'Exter Qual', 
 'Heating QC', 

#  離散型
 'Year Remod/Add',
 'Full Bath',
 'Half Bath',
 'Bedroom AbvGr',
 ]].values

X_test = new_test[[
 #名義尺度（ダミー変数で変更を施した）
 'Bldg Type_2fmCon',
 'Exterior 1st_VinylSd',
 'Electrical_FuseF',
#  連続型
 'Lot Area',  
 'Total Bsmt SF', 
 '1st Flr SF', 
 '2nd Flr SF', 
 'Gr Liv Area', 
 'Open Porch SF', 
#  順序型
 'Lot Shape', 
 'Exter Qual', 
 'Heating QC', 

#  離散型
 'Year Remod/Add',
 'Full Bath',
 'Half Bath',
 'Bedroom AbvGr',
 ]].values

y = train['SalePrice'].values

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 変数の標準化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# X と y のスケーリング
X_scaled = scaler_X.fit_transform(X)
X_test_scaled = scaler_X.transform(X_test)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))  # yも2D配列に変換してスケーリング

# モデルの学習
svr = SVR(kernel='rbf', C=1.0, gamma='scale')
svr.fit(X_scaled, y_scaled)

# 予測
y_pred_scaled = svr.predict(X_test_scaled)

# 予測結果を元のスケールに戻す
my_prediction = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()  # 予測結果も逆変換
print(my_prediction)

# csvファイルの作成
# 行単位でデータをまとめる
index = np.array(test['index']).astype(int)
data = list(zip(index, my_prediction))

# 列名無のDataFrameを作成
my_solution = pd.DataFrame(data)
my_solution.to_csv('my_tree_one.csv', index=False, header=False)
