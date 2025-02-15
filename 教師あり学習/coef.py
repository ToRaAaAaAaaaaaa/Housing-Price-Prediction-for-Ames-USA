import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler

# データの準備
X = np.random.rand(100, 5)  # 100行5列のランダムデータ
y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(100) * 0.1  # ターゲット

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徴量の標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 線形回帰
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

print("Linear Regression Coefficients:", lr.coef_)

# Lasso回帰
lasso = Lasso(alpha=0.1)  # 正則化パラメータαを調整
lasso.fit(X_train_scaled, y_train)

print("Lasso Regression Coefficients:", lasso.coef_)

# ゼロでない特徴量の確認
important_features = np.where(lasso.coef_ != 0)[0]
print("Important features (Lasso):", important_features)
