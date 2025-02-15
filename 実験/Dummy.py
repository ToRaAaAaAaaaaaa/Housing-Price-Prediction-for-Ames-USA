import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

train = pd.read_csv(r'C:\Users\itato\OneDrive\デスクトップ\機械学習\signate\【第53回_Beginner限定コンペ】アメリカの都市エイムズの住宅価格予測\csv\train.csv')
test = pd.read_csv(r'C:\Users\itato\OneDrive\デスクトップ\機械学習\signate\【第53回_Beginner限定コンペ】アメリカの都市エイムズの住宅価格予測\csv\test.csv')

# ダミー変数を作成
def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name, drop_first=False)  # drop_first=Falseですべてのカテゴリーを保持
    dummies = dummies.applymap(lambda x: 1 if x > 0 else 0)  # 値を1か0に変換
    df = pd.concat([df, dummies], axis=1)
    return df

# train と test にダミー変数を追加
train = create_dummies(train, 'Land Contour')
test = create_dummies(test, 'Land Contour')

# 結果を確認
print(train.head())
print(test.head())

import pandas as pd

# 既に加工済みのデータフレームを train と仮定します

# 新しいファイルパスを指定
output_path = r'C:\Users\itato\OneDrive\デスクトップ\機械学習\signate\【第53回_Beginner限定コンペ】アメリカの都市エイムズの住宅価格予測\csv\new_train.csv'

# データフレームをCSVとして保存
train.to_csv(output_path, index=False, encoding='utf-8')

print(f"新しい train.csv が保存されました: {output_path}")