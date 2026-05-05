"""
核心目標：使用 optuna 調整參數
"""
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score

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

# Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
        'random_state': RANDOM_STATE
    }

    # 5 折
    kFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
    
    f1_scores = []

    # K-Fold 迴圈
    for train_idx, val_idx in kFold.split(x_train, y_train):
        # 取得此折索引數據
        x_train_fold, x_value_fold = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_train_fold, y_value_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = RandomForestClassifier(**params)
        model.fit(x_train_fold, y_train_fold)

        # 預測驗證集並計算 Macro F1
        preds = model.predict(x_value_fold)
        score = f1_score(y_value_fold, preds, average='macro', zero_division=0)
        f1_scores.append(score)

    return np.mean(f1_scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"\nbest 參數組合: {study.best_params}")