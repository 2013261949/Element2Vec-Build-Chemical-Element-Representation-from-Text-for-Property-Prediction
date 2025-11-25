import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_experiment(model_name, N, X_train, X_test, y_train, y_test):
    """
    Run a complete feature selection, model training, and evaluation process for a specified number of features N.

    Args:
        N (int): Numbers of the most important dimensions
        X_train, X_test, y_train, y_test: the dataset splited already

    Returns:
        dict: dict includes four metrics: mae, mse, rmse, r2
    """
    
    if model_name == 'lr':
        feature_selection_model = LinearRegression()
        prediction_model = LinearRegression()

    elif model_name == 'xgboost':
        
        xgb_params = {
            'objective': 'reg:squarederror', 'n_estimators': 50, 'max_depth': 3,
            'learning_rate': 0.05, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42
        }
        feature_selection_model = xgb.XGBRegressor(**xgb_params)
        prediction_model = xgb.XGBRegressor(**xgb_params)

    elif model_name == 'mlp':
        mlp_params = {
            'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam',
            'alpha': 0.0001, 'max_iter': 500, 'random_state': 42,
            'early_stopping': False
        }
        feature_selection_model = MLPRegressor(**mlp_params)
        prediction_model = MLPRegressor(**mlp_params)
    
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    
    feature_selection_model.fit(X_train, y_train)
    
    if model_name == 'lr':
        importances = np.abs(feature_selection_model.coef_)
    
    elif model_name == 'xgboost':
        importances = feature_selection_model.feature_importances_

    elif model_name == 'mlp':
        # print(f"    Calculating permutation importance for N={N}...")
        perm_result = permutation_importance(
            feature_selection_model, X_train, y_train,
            n_repeats=5, random_state=42, n_jobs=-1
        )
        importances = perm_result.importances_mean
    
    sorted_feature_indices = np.argsort(importances)[::-1]
    top_N_features_indices = sorted_feature_indices[:N]

    X_train_selected = X_train[:, top_N_features_indices]
    X_test_selected = X_test[:, top_N_features_indices]

    prediction_model.fit(X_train_selected, y_train)

    y_pred = prediction_model.predict(X_test_selected)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}