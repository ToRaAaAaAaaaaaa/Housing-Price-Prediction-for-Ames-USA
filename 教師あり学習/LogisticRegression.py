import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

home = os.path.expanduser("~")
new_train_path = os.path.join(home, "OneDrive", "デスクトップ", "機械学習", "signate", "HousingPrediction", "csv", "new_train.csv")
new_train = pd.read_csv(new_train_path)

X = new_train[[
 #名義尺度（ダミー変数で変更を施した）
 'MS SubClass_30',
 'MS SubClass_50',
 'MS SubClass_60',
 'MS SubClass_70',
 'MS SubClass_80',
 'MS SubClass_85',
 'MS SubClass_90',
 'MS SubClass_120',
 'MS SubClass_160',
 'MS SubClass_180',
 'MS SubClass_190',
 'MS Zoning_RL',
 'MS Zoning_RM',
 'Land Contour_Lvl',
 'Lot Config_CulDSac',
 'Lot Config_Inside',
 'Neighborhood_BrDale',
 'Neighborhood_BrkSide',
 'Neighborhood_ClearCr',
 'Neighborhood_CollgCr',
 'Neighborhood_Crawfor',
 'Neighborhood_Edwards',
 'Neighborhood_Gilbert',
 'Neighborhood_IDOTRR',
 'Neighborhood_MeadowV',
 'Neighborhood_Mitchel',
 'Neighborhood_NAmes',
 'Neighborhood_NPkVill',
 'Neighborhood_NWAmes',
 'Neighborhood_NoRidge',
 'Neighborhood_NridgHt',
 'Neighborhood_OldTown',
 'Neighborhood_Sawyer',
 'Neighborhood_SawyerW',
 'Neighborhood_Somerst',
 'Neighborhood_StoneBr',
 'Neighborhood_Timber',
 'Bldg Type_2fmCon',
 'Bldg Type_Duplex',
 'Bldg Type_Twnhs',
 'Bldg Type_TwnhsE',
 'House Style_1Story',
 'House Style_2Story',
 'House Style_SFoyer',
 'House Style_SLvl',
 'Roof Style_Hip',
 'Exterior 1st_CemntBd',
 'Exterior 1st_HdBoard',
 'Exterior 1st_MetalSd',
 'Exterior 1st_Plywood',
 'Exterior 1st_Stucco',
 'Exterior 1st_VinylSd',
 'Exterior 1st_Wd Sdng',
 'Foundation_CBlock',
 'Foundation_PConc',
 'Foundation_Slab',
 'Central Air_Y',
 'Electrical_FuseF',
 'Electrical_SBrkr',
 'Paved Drive_Y',
 'Sale Type_WD ',
 'Sale Condition_Partial', 
#  連続型
 'Lot Area',  
 'Total Bsmt SF', 
 '1st Flr SF', 
 '2nd Flr SF', 
 'Gr Liv Area', 
 'Garage Area', 
 'Wood Deck SF', 
 'Open Porch SF', 
#  順序型
 'Lot Shape', 
 'Overall Qual', 
 'Overall Cond', 
 'Exter Qual', 
 'Heating QC', 
 'Kitchen Qual', 
#  離散型
 'Year Built',
 'Year Remod/Add',
 'Full Bath',
 'Half Bath',
 'Bedroom AbvGr',
 'Kitchen AbvGr',
 'TotRms AbvGrd',
 'Fireplaces',
 'Garage Cars',
 'Mo Sold',
 'Yr Sold',
 ]].values

y = new_train['SalePrice'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# C=1
logreg = LogisticRegression(C=1.0, max_iter=1000)
logreg.fit(X_train, y_train)
print('Training set score: {:.3f}'.format(logreg.score(X_train, y_train)))
print('Test set score: {:.3f}'.format(logreg.score(X_test, y_test)))

# C=100
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print('Training set score: {:.3f}'.format(logreg100.score(X_train, y_train)))
print('Test set score: {:.3f}'.format(logreg100.score(X_test, y_test)))

# C=0.01
logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print('Training set score: {:.3f}'.format(logreg001.score(X_train, y_train)))
print('Test set score: {:.3f}'.format(logreg001.score(X_test, y_test)))

# そもそも線形分離問題を解くためのモジュールのため棄却