import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
import sys

m = int(sys.argv[1])
filltype = int(sys.argv[2])
selecttype = int(sys.argv[3])
modeltype = int(sys.argv[4])

if filltype == 1:
    knn_str = "knn_by_season_and_group"
elif filltype == 2:
    knn_str = "knn_all_data"
elif filltype == 3:
    knn_str = "knn_random"

if selecttype == 1:
    sel_str = "handsome"
elif selecttype == 2:
    sel_str = "ic"
elif selecttype == 3:
    sel_str = "forest"

if modeltype == 1:
    model_str = "logistic"
elif modeltype == 2:
    model_str = "svm"
elif modeltype == 3:
    model_str = "rf"
elif modeltype == 4:
    model_str = "xgboost"

df = pd.read_csv(f"selected_csv/m_{m}_{knn_str}_{sel_str}.csv", index_col=0)
train_data_encoded = df[:-6185]
test_data_encoded = df[-6185:].reset_index(drop=True).drop(columns="home_team_win")

X = train_data_encoded.drop(columns=["home_team_win"])  # Drop target column
y = train_data_encoded["home_team_win"]  # Target column

e_in_list, e_val_list, e_out_list = [], [], []

best_model = None
best_e_val = float("inf")

for exp in range(5):
    # Perform train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if modeltype == 1:
        # 定義 Logistic Regression 模型和參數網格
        logistic_model = LogisticRegression(max_iter=10000)
        param_grid = {
            "C": [0.01, 0.05, 0.1],  # 正則化參數
            "penalty": ["l1", "l2"],  # 正則化方式
            "solver": ["liblinear"],  # 優化算法
        }

        # 使用 GridSearchCV 進行參數選擇
        grid_search = GridSearchCV(
            logistic_model, param_grid, scoring="accuracy", cv=5, verbose=1, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # 獲取最佳模型
        best_logistic_model = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")

        # 驗證集預測
        train_predictions = best_logistic_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        print(f"Train Accuracy: {train_accuracy:.4f}")

        val_predictions = best_logistic_model.predict(X_val)
        validation_accuracy = accuracy_score(y_val, val_predictions)
        print(f"Validation Accuracy: {validation_accuracy:.4f}")

        # 用全量資料進行訓練
        y_full_train = train_data_encoded["home_team_win"]
        X_full_train = train_data_encoded.drop(columns=["home_team_win"])

        # 使用最佳參數定義 Logistic Regression 模型
        final_logistic_model = LogisticRegression(
            C=grid_search.best_params_["C"],
            penalty=grid_search.best_params_["penalty"],
            solver=grid_search.best_params_["solver"],
            max_iter=10000,
        )

        # 訓練模型
        final_logistic_model.fit(X_full_train, y_full_train)

        # 對 test_data_encoded 進行預測
        test_predictions = final_logistic_model.predict(test_data_encoded)

        # 將結果存入 DataFrame，符合指定格式
        final_model = final_logistic_model
        test_results = pd.DataFrame(
            {
                "id": np.arange(len(test_predictions)),
                "home_team_win": test_predictions.astype(bool),
            }
        )

    elif modeltype == 3:
        rf_model = RandomForestClassifier()
        param_grid = {
            "n_estimators": [10, 20, 30],  # 樹的數量
            "max_depth": [1, 2, 3, 4, 5],  # 樹的最大深度
            "min_samples_split": [15],  # 節點分裂所需的最小樣本數
            "min_samples_leaf": [10],  # 葉子節點所需的最小樣本數
            "criterion": ["gini", "entropy"],  # 分裂準則
            "max_features": [1, 3, 5],  # 每次分裂時考慮的最大特徵數
            "class_weight": [None, "balanced"],  # 類別權重
        }

        # 使用 GridSearchCV 進行參數選擇
        grid_search = GridSearchCV(
            rf_model, param_grid, scoring="accuracy", cv=5, verbose=1, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # 獲取最佳模型
        best_rf_model = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")

        # 驗證集預測
        train_predictions = best_rf_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        print(f"Train Accuracy: {train_accuracy:.4f}")

        val_predictions = best_rf_model.predict(X_val)
        validation_accuracy = accuracy_score(y_val, val_predictions)
        print(f"Validation Accuracy: {validation_accuracy:.4f}")

        # 用全量資料進行訓練
        y_full_train = train_data_encoded["home_team_win"]
        X_full_train = train_data_encoded.drop(columns=["home_team_win"])

        # 定義 RandomForest 模型和最佳參數（根據之前的 GridSearch 結果）
        final_rf_model = RandomForestClassifier(
            n_estimators=grid_search.best_params_["n_estimators"],
            max_depth=grid_search.best_params_["max_depth"],
            min_samples_split=grid_search.best_params_["min_samples_split"],
            min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
            criterion=grid_search.best_params_["criterion"],
            max_features=grid_search.best_params_["max_features"],
            class_weight=grid_search.best_params_["class_weight"],
        )

        # 訓練模型
        final_rf_model.fit(X_full_train, y_full_train)

        # 對 test_data_encoded 進行預測
        test_predictions = final_rf_model.predict(test_data_encoded)

        # 將結果存入 DataFrame，符合指定格式
        final_model = final_rf_model
        test_results = pd.DataFrame(
            {
                "id": np.arange(len(test_predictions)),
                "home_team_win": test_predictions.astype(bool),
            }
        )

    true_value = pd.read_csv("dataset/same_season_test_label.csv")
    mismatched_rows = test_results["home_team_win"] != true_value["home_team_win"]
    mismatched_count = mismatched_rows.sum()

    e_in = train_accuracy
    e_val = validation_accuracy
    e_out = 1 - mismatched_count / len(true_value)
    print(f"Number of mismatched rows: {mismatched_count}")
    print(f"Accuracy: {e_out}")

    e_in_list.append(e_in)
    e_val_list.append(e_val)
    e_out_list.append(e_out)

    if e_val < best_e_val:
        best_model = final_model


e_in_avg = sum(e_in_list) / len(e_in_list)
e_val_avg = sum(e_val_list) / len(e_val_list)
e_out_avg = np.mean(e_out_list)

error_data = {
    "E_in": e_in_list,
    "E_val": e_val_list,
    "E_out": e_out_list,
}
results_df = pd.DataFrame(error_data)
results_df.to_csv(f"models_new/m_{m}_{knn_str}_{sel_str}_{model_str}.csv", index=False)

# Save the trained model to a pickle file
with open(f"models_new/m_{m}_{knn_str}_{sel_str}_{model_str}.pkl", "wb") as f:
    pickle.dump(final_model, f)
