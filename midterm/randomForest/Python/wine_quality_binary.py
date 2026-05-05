"""
核心目標： 簡化為「好酒/壞酒」，讓此模型具商業價值

更動與參數：
    將多分類改為二元分類（>=6 為好酒）
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

x = data.drop([TARGET_COL, 'Id'], axis=1)

# 二元分類：6, 7, 8 -> 1 (Good); 3, 4, 5 -> 0 (Bad)
y = data[TARGET_COL].apply(lambda x: 1 if x >= 6 else 0)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15)

v3_binary = RandomForestClassifier(n_estimators=743, class_weight='balanced', max_depth=20, min_samples_split= 9, random_state=RANDOM_STATE)
v3_binary.fit(x_train, y_train)

y_pred = v3_binary.predict(x_test)
print("v3 binary：")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

# 混淆矩陣
c_m = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(c_m, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.title('V3 Binary Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()