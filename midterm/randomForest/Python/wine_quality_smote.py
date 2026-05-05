"""
核心目標： 使用 SMOTE 技術，解決資料集不平衡問題
"""
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE

DATA_PATH = 'WineQT.csv'
TARGET_COL = 'quality'
RANDOM_STATE = 15

try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print("WineQT.csv not found.")
    exit()

x = data.drop(['quality', 'Id'], axis=1)
y = data['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)

# 對全體訓練集做 SMOTE
smote_final = SMOTE(random_state=RANDOM_STATE, k_neighbors=2)
x_result, y_result = smote_final.fit_resample(x_train, y_train)

v5_smote = RandomForestClassifier(n_estimators=743, class_weight='balanced', max_depth=20, min_samples_split= 9, random_state=RANDOM_STATE)
v5_smote.fit(x_result, y_result)

y_pred = v5_smote.predict(x_test)
print("\nv5 SMOTE：")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

# 混淆矩陣
c_m = confusion_matrix(y_test, y_pred)
labels = sorted(y_test.unique())
plt.figure(figsize=(10, 8))
sns.heatmap(c_m, annot=True, fmt='d', cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
plt.title('V5 SMOTE', fontsize=16)
plt.show()