import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 15
np.random.seed(RANDOM_STATE)

df = pd.read_csv("WineQT.csv").set_index("Id")
X = df.drop(columns=["quality"])
Y = df["quality"]
# new feature
X["alc_vol_acid_ratio"] = X["alcohol"] / (X["volatile acidity"] + 1e-5)
X["sulfur_ratio"] = X["free sulfur dioxide"] / (X["total sulfur dioxide"] + 1e-5)
X["alc_sugar_ratio"] = X["alcohol"] / (X["residual sugar"] + 1e-5)
X["sulphate_alc"] = X["sulphates"] * X["alcohol"]
X["density_alc"] = X["density"] / (X["alcohol"] + 1e-5)
X["acid_ph_interact"] = X["fixed acidity"] * X["pH"]


QUALITY_CLASSES = [3, 4, 5, 6, 7, 8]  
OR_THR = [4, 5, 6, 7, 8]
N_CLASSES = 6


def smote(X_tr: pd.DataFrame, Y_tr: pd.Series, target: int = 100) -> tuple:
    counts = Y_tr.value_counts()
    strategy = {cls: target for cls, cnt in counts.items() if cnt < target}
    if not strategy:
        return X_tr.values, Y_tr.values
    

    min_cnt = counts.min()
    k = min(3, int(min_cnt) - 1)
    if k < 1:
        return X_tr.values, Y_tr.values
    try:
        sm = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=RANDOM_STATE)
        Xr, Yr = sm.fit_resample(X_tr, Y_tr)
        return Xr, Yr
    except Exception:
        return X_tr.values, Y_tr.values


def ordinal_predict(models, X, return_prob = False):

    raw = np.column_stack([models[t].predict_proba(X)[:, 1] for t in OR_THR])
    ir = IsotonicRegression(increasing=False)
    corr = np.array([ir.fit_transform(np.arange(5), row) for row in raw])

    prob = np.zeros((len(X), N_CLASSES))
    prob[:, 0] = 1 - corr[:, 0]
    for i in range(1, 5):
        prob[:, i] = corr[:, i-1] - corr[:, i]
    prob[:, 5] = corr[:, 4]
    prob = np.clip(prob, 0, 1)

    if return_prob:
        return prob
    return np.array(QUALITY_CLASSES)[np.argmax(prob, axis=1)]


def train_ordinal_models(X_tr, Y_tr, X_va, Y_va, params: dict) -> dict: 
    models = {}
    for i in OR_THR:
        y_tr_bin = (Y_tr >= i).astype(int)
        y_va_bin = (Y_va >= i).astype(int)
        neg = (y_tr_bin == 0).sum()
        pos = (y_tr_bin == 1).sum()
        p = {**params, 'scale_pos_weight': neg/(pos + 1e-5)}
        clf = lgb.LGBMClassifier(**p)
        clf.fit(X_tr, y_tr_bin, eval_set=[(X_va, y_va_bin)], callbacks=[lgb.early_stopping(20, verbose=False),lgb.log_evaluation(-1)])
        models[i] = clf
    return models


OUTER_K = 5
INNER_K = 3
N_TRIALS = 25 # optuna trials

outer_skf = StratifiedKFold(n_splits=OUTER_K, shuffle=True, random_state=RANDOM_STATE)
outer_macro_f1 = []
outer_weighted_f1 = []
outer_reports = []
best_params_list = []

X_arr = X.values
Y_arr = Y.values

for outer_fold, (train_idx, test_idx) in enumerate(outer_skf.split(X_arr, Y_arr), 1):
    print(f"fold: {outer_fold}/{OUTER_K}")

    X_outer_tr = X.iloc[train_idx]
    Y_outer_tr = Y.iloc[train_idx]
    X_outer_te = X.iloc[test_idx]
    Y_outer_te = Y.iloc[test_idx]
    
    inner_skf = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=RANDOM_STATE)

    def inner_objective(trial):
        param = {
            "objective" : "binary",
            "verbosity" : -1,
            "boosting_type" : "gbdt",
            "learning_rate" :trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "num_leaves" : trial.suggest_int("num_leaves", 15, 50),
            "max_depth" : trial.suggest_int("max_depth", 3, 7),
            "min_child_samples" : trial.suggest_int("min_child_samples", 15, 60),
            "subsample" : trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_alpha" : trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda" : trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "min_split_gain" : trial.suggest_float("min_split_gain", 0.0, 0.5),
            "n_estimators" : 300,
            "random_state" : RANDOM_STATE,
        }

        inner_f1 = []
        for tr_i, va_i in inner_skf.split(X_outer_tr, Y_outer_tr):
            Xi_tr = X_outer_tr.iloc[tr_i]
            Yi_tr = Y_outer_tr.iloc[tr_i]
            Xi_va = X_outer_tr.iloc[va_i]
            Yi_va = Y_outer_tr.iloc[va_i]

            Xi_tr_sm, Yi_tr_sm = smote(Xi_tr, Yi_tr, target=80)

            models = train_ordinal_models(Xi_tr_sm, Yi_tr_sm, Xi_va.values, Yi_va.values,param)
            y_pred = ordinal_predict(models, Xi_va.values)
            inner_f1.append(f1_score(Yi_va, y_pred, average='macro', zero_division=0))
        return np.mean(inner_f1)

    inner_study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    inner_study.optimize(inner_objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_p = {
        **inner_study.best_params,
        'objective' : 'binary',
        'verbosity' : -1,
        'boosting_type': 'gbdt',
        'n_estimators' : 600,
        'random_state' : RANDOM_STATE,
    }
    best_params_list.append(inner_study.best_params)
    print(f"best inner macro F1: {inner_study.best_value:.4f}")

    X_tr_sm, Y_tr_sm = smote(X_outer_tr, Y_outer_tr, target=100)
    final_models = train_ordinal_models(X_tr_sm, Y_tr_sm,X_outer_te.values, Y_outer_te.values,best_p)

    y_pred_outer = ordinal_predict(final_models, X_outer_te.values)

    macro = f1_score(Y_outer_te, y_pred_outer, average="macro", zero_division=0)
    weighted = f1_score(Y_outer_te, y_pred_outer, average="weighted", zero_division=0)

    outer_macro_f1.append(macro)
    outer_weighted_f1.append(weighted)
    outer_reports.append(classification_report(Y_outer_te, y_pred_outer,zero_division=0, output_dict=True))
    print(f" out macro F1: {macro:.4f} | weighted F1: {weighted:.4f}")


macro_arr = np.array(outer_macro_f1)
weighted_arr = np.array(outer_weighted_f1)

print(f"Macro F1: {macro_arr.mean():.4f} ± {macro_arr.std():.4f}")
print(f"Weighted F1: {weighted_arr.mean():.4f} ± {weighted_arr.std():.4f}")
print(f"fold: {[f'{v:.4f}' for v in macro_arr]}")
print()
print(f"  {"Quality":>8} | {"Precision":>9} | {"Recall":>9} | {"F1":>9} | {"Support":>7}")
print()

for q in [str(c) for c in QUALITY_CLASSES]:
    p_vals, r_vals, f_vals, s_vals = [], [], [], []
    for rep in outer_reports:
        if q in rep:
            p_vals.append(rep[q]["precision"])
            r_vals.append(rep[q]["recall"])
            f_vals.append(rep[q]["f1-score"])
            s_vals.append(rep[q]["support"])
    if p_vals:
        print(f"  {q:>8} | "
              f"{np.mean(p_vals):>8.3f}  | "
              f"{np.mean(r_vals):>8.3f}  | "
              f"{np.mean(f_vals):>8.3f}  | "
              f"{np.mean(s_vals):>7.1f}")

avg_params = {}
for key in best_params_list[0]:
    vals = [bp[key] for bp in best_params_list]
    if isinstance(vals[0], int):
        avg_params[key] = int(np.round(np.mean(vals)))
    else: 
        avg_params[key] = np.mean(vals)

final_params = {
    **avg_params,
    "objective" : "binary",
    "verbosity" : -1,
    "boosting_type" : "gbdt",
    "n_estimators" : 600,
    "random_state" : RANDOM_STATE,
}

print(f"paras: {avg_params}")

X_all_sm, Y_all_sm = smote(X, Y, target=120)
X_all_sm = pd.DataFrame(X_all_sm, columns=X.columns)
Y_all_sm = pd.Series(Y_all_sm)

final_all_models = {}
for t in OR_THR:
    y_bin = (Y_all_sm >= t).astype(int)
    neg = (y_bin == 0).sum()
    pos = (y_bin == 1).sum()
    p = {**final_params, 'scale_pos_weight': neg / (pos + 1e-5)}
    clf = lgb.LGBMClassifier(**p)
    clf.fit(X_all_sm, y_bin, callbacks=[lgb.log_evaluation(-1)])
    final_all_models[t] = clf

imp_df = pd.DataFrame({t: pd.Series(m.feature_importances_, index=X_all_sm.columns) for t, m in final_all_models.items()})
avg_imp = imp_df.mean(axis=1).sort_values(ascending=False)

print("gain avg: ")
print(avg_imp.to_string())

print("Result")
print(f"Macro F1: {macro_arr.mean():.4f} ± {macro_arr.std():.4f}")
print(f"Weighted F1: {weighted_arr.mean():.4f} ± {weighted_arr.std():.4f}")
