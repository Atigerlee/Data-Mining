import pandas as pd
import numpy as np
import math
import optuna  

def predict(train_x_np, train_y_np, test_row_np, k, d_func, is_weighted):
    if d_func == "e":
        distances = np.linalg.norm(train_x_np - test_row_np, axis=1)
    elif d_func == "m":
        distances = np.sum(np.abs(train_x_np - test_row_np), axis=1)    
    
    k_indices = np.argsort(distances)[:k]
    vote = {}
    
    for idx in k_indices:
        label = train_y_np[idx]
        if is_weighted:
            weight = 1 /(distances[idx] + 1e-9)
        else:
            weight = 1
        if label in vote:
            vote[label] +=weight
        else:
            vote[label] =weight
            
    return max(vote, key=vote.get)


df = pd.read_csv("WineQT.csv")
df = df.set_index("Id")

df_train_full = df.sample(frac=0.8, random_state=15)
df_test = df.drop(df_train_full.index)

# split
df_train = df_train_full.sample(frac=0.8, random_state=15)
df_val = df_train_full.drop(df_train.index)

train_x_raw = df_train.drop(columns=["quality"])
train_y = df_train["quality"]
val_x_raw = df_val.drop(columns=["quality"])
val_y = df_val["quality"]
test_x_raw = df_test.drop(columns=["quality"])
test_y = df_test["quality"]

train_y_np = train_y.values
val_y_np = val_y.values
test_y_np = test_y.values

mean = train_x_raw.mean(axis=0)
std = train_x_raw.std(axis=0)
train_x_zscore = ((train_x_raw - mean) /std).values
val_x_zscore = ((val_x_raw - mean) /std).values
test_x_zscore = ((test_x_raw - mean) /std).values

col_min = train_x_raw.min(axis=0)
col_max = train_x_raw.max(axis=0)
train_x_minmax = ((train_x_raw - col_min) /(col_max - col_min)).values
val_x_minmax = ((val_x_raw - col_min) /(col_max - col_min)).values
test_x_minmax = ((test_x_raw - col_min) /(col_max - col_min)).values


max_k = int(math.sqrt(len(df_train)))

# optuna
def objective(trial):
    k = trial.suggest_int("k", 1, max_k, step=2)  
    scale_name = trial.suggest_categorical("scale_method", ["Z-Score", "Min-Max"])
    d_code = trial.suggest_categorical("distance_metric", ["e", "m"])
    is_weighted = trial.suggest_categorical("is_weighted", [True, False])

    if scale_name == "Z-Score":
        current_train_x, current_val_x = train_x_zscore, val_x_zscore
    else:
        current_train_x, current_val_x = train_x_minmax, val_x_minmax

    preds = []
    for i in range(len(current_val_x)):
        a = predict(current_train_x, train_y_np, current_val_x[i], k, d_code, is_weighted=is_weighted)
        preds.append(a)
    acc = sum(p == t for p, t in zip(preds, val_y_np)) /len(val_y_np)

    classes = np.unique(val_y_np)
    f1_scores = []
    for c in classes:
        tp = sum((p == c) and (t == c) for p, t in zip(preds, val_y_np))
        fp = sum((p == c) and (t != c) for p, t in zip(preds, val_y_np))
        fn = sum((p != c) and (t == c) for p, t in zip(preds, val_y_np))
        
        precision = tp /(tp + fp) if(tp + fp) > 0 else 0
        recall = tp /(tp + fn) if(tp + fn) > 0 else 0
        
        if precision + recall > 0:
            f1 = 2 *(precision * recall)/(precision + recall)
        else:
            f1 = 0
        f1_scores.append(f1)
        
    macro_f1 = sum(f1_scores) /len(f1_scores)

    trial.set_user_attr("accuracy", acc)
    return macro_f1

# find
study = optuna.create_study(direction="maximize")  
study.optimize(objective, n_trials=30)  

print()
print("Best Validation Model:")
print(f"  Best Macro F1-Score: {study.best_value:.4f}")
print(f"  Accuracy: {study.best_trial.user_attrs["accuracy"]:.4f}") 
print()
print("Best Parameters:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")
print()

best_params = study.best_params
best_k = best_params["k"]
best_scale = best_params["scale_method"]
best_d_code = best_params["distance_metric"]
best_is_weighted = best_params["is_weighted"]


if best_scale == "Z-Score":
    final_train_x, final_test_x = train_x_zscore, test_x_zscore
else:
    final_train_x, final_test_x = train_x_minmax, test_x_minmax

best_predictions = []
for i in range(len(final_test_x)):
    res = predict(final_train_x, train_y_np, final_test_x[i], best_k, best_d_code, best_is_weighted)
    best_predictions.append(res)

conf_matrix = pd.crosstab(pd.Series(test_y_np, name="Actual"), pd.Series(best_predictions, name="Predicted"))

print("Confusion Matrix")
print(conf_matrix)
print()
# print("Data Split Distribution:")
# print(f"Train size: {len(df_train)}, Val size: {len(df_val)}, Test size: {len(df_test)}")
