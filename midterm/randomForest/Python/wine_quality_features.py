"""
核心目標： 使用特徵工程建立新特徵，加上混淆矩陣查看模型誤判狀況

更動與參數：
    將 [酒精 / 揮發性酸度] 作為新特徵，並且去除雜訊 [residual sugar]
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = 'WineQT.csv'
TARGET_COL = 'quality'
RANDOM_STATE = 15

try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print("WineQT.csv not found.")
    exit()

# 特徵工程
data['alc_acid_ratio'] = data['alcohol'] / data['volatile acidity']

x = data.drop(['quality', 'Id', 'residual sugar'], axis=1)
y = data['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)

v4_features = RandomForestClassifier(n_estimators=743, class_weight='balanced', max_depth=20, min_samples_split= 9, random_state=RANDOM_STATE)
v4_features.fit(x_train, y_train)

y_pred = v4_features.predict(x_test)
print("v4 features：")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='RdPu',
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.title('Confusion Matrix (v4 Features)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()