import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error

home = os.path.expanduser("~")
train_path = os.path.join(home, "OneDrive", "デスクトップ", "機械学習", "signate", "HousingPrediction", "csv", "train.csv")
train = pd.read_csv(train_path)

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

'''
順序尺度の変換
'''

# Lot Shape
train.loc[train["Lot Shape"] == "Reg", "Lot Shape"] = 4
train.loc[train["Lot Shape"] == "IR1", "Lot Shape"] = 3
train.loc[train["Lot Shape"] == "IR2", "Lot Shape"] = 2
train.loc[train["Lot Shape"] == "IR3", "Lot Shape"] = 1

# Exter Qual
train.loc[train["Exter Qual"] == "Ex", "Exter Qual"] = 5
train.loc[train["Exter Qual"] == "Gd", "Exter Qual"] = 4
train.loc[train["Exter Qual"] == "TA", "Exter Qual"] = 3
train.loc[train["Exter Qual"] == "Fa", "Exter Qual"] = 2
train.loc[train["Exter Qual"] == "Po", "Exter Qual"] = 1

# Heating QC
train.loc[train["Heating QC"] == "Ex", "Heating QC"] = 5
train.loc[train["Heating QC"] == "Gd", "Heating QC"] = 4
train.loc[train["Heating QC"] == "TA", "Heating QC"] = 3
train.loc[train["Heating QC"] == "Fa", "Heating QC"] = 2
train.loc[train["Heating QC"] == "Po", "Heating QC"] = 1

# Kitchen Qual
train.loc[train["Kitchen Qual"] == "po", "Kitchen Qual"] = 1
train.loc[train["Kitchen Qual"] == "Fa", "Kitchen Qual"] = 2
train.loc[train["Kitchen Qual"] == "TA", "Kitchen Qual"] = 3
train.loc[train["Kitchen Qual"] == "Gd", "Kitchen Qual"] = 4
train.loc[train["Kitchen Qual"] == "Ex", "Kitchen Qual"] = 5

# 新しいファイルパスを指定
output_path = r'C:\Users\itato\OneDrive\デスクトップ\機械学習\signate\【第53回_Beginner限定コンペ】アメリカの都市エイムズの住宅価格予測\csv\new_train.csv'

# データフレームをCSVとして保存
train.to_csv(output_path, index=False, encoding='utf-8')

# 特徴量の決定
new_train = pd.read_csv(r'C:\Users\itato\OneDrive\デスクトップ\機械学習\signate\【第53回_Beginner限定コンペ】アメリカの都市エイムズの住宅価格予測\csv\new_train.csv')
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

y = train['SalePrice'].values

# 変数の標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# モデルの学習
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0, test_size=0.3)
lasso = Lasso(alpha=0.01, max_iter=100000)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)

# MSEの計算
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')

# MAEの計算
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# 評価
print('Training set score: {:.2f}'.format(lasso.score(X_train, y_train)))
print('Test set score: {:.2f}'.format(lasso.score(X_test, y_test)))

# Mean Squared Error (MSE): 748054034.6159981
# Mean Absolute Error (MAE): 17256.015736610098
# Training set score: 0.20
# Test set score: 0.15

important_features = np.where(lasso.coef_ != 0)[0]
print(important_features)
