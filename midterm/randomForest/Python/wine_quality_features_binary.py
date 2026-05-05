"""
核心目標： 融合特徵工程與二分法，觀察模型是否進步
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score

DATA_PATH = 'WineQT.csv'
TARGET_COL = 'quality'
RANDOM_STATE = 15

try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print("WineQT.csv not found.")
    exit()

data['alc_acid_ratio'] = data['alcohol'] / data['volatile acidity']
x = data.drop([TARGET_COL, 'Id', 'residual sugar'], axis=1)
y = data[TARGET_COL].apply(lambda x: 1 if x >= 6 else 0)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15)

v4_features_binary = RandomForestClassifier(n_estimators=743, class_weight='balanced', max_depth=12, min_samples_split=9, random_state=15)
v4_features_binary.fit(x_train, y_train)

y_pred = v4_features_binary.predict(x_test)
print("v4 features (binary)：")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

# 取得預測機率
y_scores = v4_features_binary.predict_proba(x_test)[:, 1]

# PRC 曲線
precision, recall, _ = precision_recall_curve(y_test, y_scores)
averagePrecisonScore = average_precision_score(y_test, y_scores)

# 繪圖
plt.figure(figsize=(8, 6))
plt.step(recall, precision, color='purple', alpha=0.8, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='purple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'V4 PRC Curve (Average Precision = {averagePrecisonScore:.2f})')
plt.grid(alpha=0.3)
plt.show()

print(f"v4 features (binary): {averagePrecisonScore:.4f}")