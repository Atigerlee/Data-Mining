"""
核心目標： 建立標準流程
更動與參數： 僅做最基本的資料清洗，使用 Scikit-learn 預設參數
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_PATH = 'WineQT.csv'
TARGET_COL = 'quality'
RANDOM_STATE = 15

# 讀取資料
try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print("WineQT.csv not found.")
    exit()

# 特徵 (X) 目標 (y)
x = data.drop([TARGET_COL, 'Id'], axis=1)
y = data[TARGET_COL]

# 劃分資料
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)

v1_baseline = RandomForestClassifier(random_state=RANDOM_STATE)
v1_baseline.fit(x_train, y_train)

# 輸出
print("v1 baseline：")
print(classification_report(y_test, v1_baseline.predict(x_test), digits=4, zero_division=0))