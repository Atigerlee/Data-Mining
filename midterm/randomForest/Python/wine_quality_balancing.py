"""
核心目標： optuna 參數調整完，解決完全無視少數類別 3, 4, 8 分 的問題

更動與參數：{n_estimators=743, class_weight='balanced', max_depth=20, min_samples_split= 9}
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_PATH = 'WineQT.csv'
TARGET_COL = 'quality'
RANDOM_STATE = 15

try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print("WineQT.csv not found.")
    exit()

x = data.drop([TARGET_COL, 'Id'], axis=1)
y = data[TARGET_COL]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)

# 建立模型
v2_balancing = RandomForestClassifier(
    n_estimators=743,           # 增加樹量到 500
    class_weight='balanced',    # 加重少數類別權重
    max_depth=20,               # 限制深度
    min_samples_split=9,        # 限制最低分支樣本數
    random_state=RANDOM_STATE
)
v2_balancing.fit(x_train, y_train)

print("v2 balancing：")
print(classification_report(y_test, v2_balancing.predict(x_test), digits=4, zero_division=0))