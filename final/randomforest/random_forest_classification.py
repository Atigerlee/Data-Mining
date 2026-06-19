"""
Classification — Random Forest + Mahalanobis distance

流程：
  1. 讀取訓練集 / 測試集
  2. 將訓練集再切分為 Train / Validation
  3. 特徵縮放: StandardScaler
  4. 以 Optuna 調整隨機森林, 超參數目標: Validation 準確率
  5. 用 Validation 驗證信心閾值 0.7 的切分效果
  6. 全部訓練資料重新訓練最終模型，對測試集做預測
  7. 雙重未知偵測：
       (a) RF 最高機率 < 0.7
       (b) Mahalanobis distance > 訓練分布門檻
     只要符合任一條件即標為 Low_Confidence
  8. 未知樣本標為 Unlabeled, 輸出精簡 csv 供下一階段分群使用
     欄位: 原始特徵... | Class | confidence_score | pred_label
"""

import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 15
THRESHOLD = 0.7           # RF 信心閾值（可於第三階段回頭微調）
NOVELTY_PCT = 95.0        # Mahalanobis distance 門檻
N_OPTUNA_TRIALS = 30      # Optuna 試驗次數


# 1. 讀取資料
train_data = pd.read_csv('dry_bean_train.csv')
test_data = pd.read_csv('dry_bean_test.csv')

feature_names = [col for col in train_data.columns if col != 'Class']

X = train_data[feature_names].values
y = train_data['Class'].values
X_test = test_data[feature_names].values
y_test = test_data['Class'].values        # 用於第三階段評估 不參與訓練


# 2. 把訓練集再切成 Train / Validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
)


# 3. 特徵縮放, scaler 只用 Train 來擬合
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# 4. 用 Optuna 找隨機森林最好的超參數
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': RANDOM_STATE, 'n_jobs': -1,
    }
    rf = RandomForestClassifier(**params)
    rf.fit(X_train_scaled, y_train)
    return accuracy_score(y_val, rf.predict(X_val_scaled))


study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
best_params = study.best_params
print(f'Best Validation accuracy: {study.best_value:.4f}')
print(f'Best Parameters: {best_params}')


# 5. 用最佳參數在 Train 上訓練, 於 Validation 上掃描信心閾值
val_rf = RandomForestClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
val_rf.fit(X_train_scaled, y_train)
val_probs = val_rf.predict_proba(X_val_scaled)
val_max_prob = val_probs.max(axis=1)
val_pred = val_rf.classes_[val_probs.argmax(axis=1)]

print('閾值  保留率(視為已知)  保留樣本的準確率')
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    kept = val_max_prob >= threshold
    retention = kept.mean()
    kept_acc = accuracy_score(y_val[kept], val_pred[kept]) if kept.any() else float('nan')
    mark = '  <-- 採用' if abs(threshold - THRESHOLD) < 1e-9 else ''
    print(f'{threshold:>4}      {retention:6.3f}            {kept_acc:6.3f}{mark}')


# 6. 全部訓練資料重新擬合 scaler 與模型，對測試集做最終預測
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)
X_test_scaled = final_scaler.transform(X_test)

rf = RandomForestClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_scaled, y)

test_probs = rf.predict_proba(X_test_scaled)
test_max_prob = test_probs.max(axis=1)
test_pred = rf.classes_[test_probs.argmax(axis=1)]


# 7. Mahalanobis distance：用訓練集的平均與共變異數 知道測試樣本離訓練分布多遠
#    距離公式 = sqrt( (x - mean)^T * inv_cov * (x - mean) )
train_mean = X_scaled.mean(axis=0)
train_cov = np.cov(X_scaled, rowvar=False)
inv_cov = np.linalg.pinv(train_cov)        # 用偽逆比較穩定


def mahalanobis(samples, mean, inv_cov):
    centered = samples - mean
    # 對每一列算二次型 (x-mean)·inv_cov·(x-mean) 再開根號
    return np.sqrt(np.sum((centered @ inv_cov) * centered, axis=1))


train_novelty = mahalanobis(X_scaled, train_mean, inv_cov)
novelty_threshold = np.percentile(train_novelty, NOVELTY_PCT)   # 門檻取自訓練分布
test_novelty = mahalanobis(X_test_scaled, train_mean, inv_cov)


# 8. 雙重未知偵測：信心太低 或 離訓練分布太遠 -> 未知
low_confidence = test_max_prob < THRESHOLD
is_novel = test_novelty > novelty_threshold
is_unknown = low_confidence | is_novel

final_labels = test_pred.copy()
final_labels[is_unknown] = 'Unlabeled'

result = test_data[feature_names].copy()    # 原始特徵
result['Class'] = y_test                    # 保留真實標籤
result['confidence_score'] = test_max_prob  # RF 對該預測的最高機率
result['pred_label'] = final_labels         # 未知樣本標為 Unlabeled

result.to_csv('rf_classification_result.csv', index=False, encoding='utf-8-sig')
joblib.dump(final_scaler, 'scaler.joblib')


# 摘要
print(f'Mahalanobis distance threshold (訓練 {NOVELTY_PCT:.0f} 百分位): {novelty_threshold:.3f}')
print(f'測試集總筆數                 : {len(X_test)}')
print(f'  其中僅 RF 低信心觸發       : {int((low_confidence & ~is_novel).sum())}')
print(f'  其中僅 高新穎度觸發        : {int((is_novel & ~low_confidence).sum())}')
print(f'  兩者皆觸發                 : {int((low_confidence & is_novel).sum())}')
print(f'High_Confidence              : {int((~is_unknown).sum())}')
print(f'Low_Confidence               : {int(is_unknown.sum())}')
high_conf_acc = accuracy_score(y_test[~is_unknown], test_pred[~is_unknown])
print(f'High_Confidence 子集的準確率 : {high_conf_acc:.4f}')
print('輸出: rf_classification_result.csv')
