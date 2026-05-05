import pandas as pd
import numpy as np
import optuna
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from optuna.samplers import TPESampler
from imblearn.over_sampling import SMOTE

df = pd.read_csv("WineQT.csv")
df = df.set_index("Id")

X = df.drop(columns=["quality"])


X["alc_vol_acid_ratio"] = X["alcohol"] / (X["volatile acidity"] + 1e-5)
X["sulfur_ratio"] = X["free sulfur dioxide"] / (X["total sulfur dioxide"] + 1e-5)
X["alc_sugar_ratio"] = X["alcohol"] / (X["residual sugar"] + 1e-5)
# X["total_acidity"] = X["fixed acidity"] + X["volatile acidity"]
# X = X.drop(columns=["density", "fixed acidity"])

Y = df["quality"]
le = LabelEncoder()
Ye = le.fit_transform(Y)
X_temp, X_test, y_temp, y_test = train_test_split(X, Ye, test_size=0.2, random_state=15, stratify=Ye)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=15, stratify=y_temp)
# sm = SMOTE(random_state=15, k_neighbors=3)
# X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

def objective(trial):
    para = {
        "objective" : "multi:softmax", 
        "num_class" : len(set(y_train)),
        "eval_metric" : "mlogloss", 
        "random_state" : 15,
        "early_stopping_rounds" : 30,
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        # hp
        "max_depth" : trial.suggest_int("max_depth",3,6),
        "learning_rate" : trial.suggest_float("learning_rate",0.01,0.1),
        "n_estimators" : 1000, # for early stop
        "subsample" : trial.suggest_float("subsample",0.6,0.9),
        "colsample_bytree" : trial.suggest_float("colsample_bytree",0.6,0.9),
        "gamma" : trial.suggest_float("gamma",0,5),
        "min_child_weight" : trial.suggest_int("min_child_weight",3,10)
        
    }

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=15)
    f1_scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_va = y_train[train_idx], y_train[val_idx]
        
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_tr)
        model = xgb.XGBClassifier(**para)
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
            sample_weight=sample_weights
        )
        
        pred = model.predict(X_va)
        f1_scores.append(f1_score(y_va, pred, average="macro"))
    # trial.set_user_attr("acc",acc)
    # trial.set_user_attr("f1",f1)
    trial.set_user_attr("best_n_estimators", model.best_iteration + 1)
    # return acc
    return np.mean(f1_scores)


# find
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=15))
study.optimize(objective, n_trials=100)
best_n_trees = study.best_trial.user_attrs["best_n_estimators"]
print(study.best_params)
print(f"early stop tree#: {best_n_trees}")
weights_temp = compute_sample_weight(class_weight='balanced', y=y_temp)
best_model = xgb.XGBClassifier(
    **study.best_params,
    objective="multi:softmax",
    n_estimators=best_n_trees,
    num_class=len(set(y_train)),
    eval_metric="mlogloss",
    random_state=15
)
# X_temp_res, y_temp_res = sm.fit_resample(X_temp, y_temp)
# best_model.fit(X_temp, y_temp) # baseline
best_model.fit(X_temp, y_temp, sample_weight=weights_temp) # class weight
# best_model.fit(X_temp_res, y_temp_res) # smote
final_pred = best_model.predict(X_test)

print(f"test set acc: {accuracy_score(y_test, final_pred):.4f}")
print(f"test set f1 Score: {f1_score(y_test, final_pred, average="macro"):.4f}")

importance = best_model.get_booster().get_score(importance_type="gain")

# plot
imp = pd.DataFrame({"feature": list(importance.keys()),"gain": list(importance.values())})
imp["gain"] = imp["gain"] /imp["gain"].sum()
imp = imp.sort_values(by="gain", ascending=False).head(10)

plt.barh(imp["feature"],imp["gain"])
plt.gca().invert_yaxis()

plt.gca().xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x*100:.1f}%")
)

plt.title("Top 10 Feature Importance(Gain)")
plt.xlabel("Importance (%)")

plt.show()
