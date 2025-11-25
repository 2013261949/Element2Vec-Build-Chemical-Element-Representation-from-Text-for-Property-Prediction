import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold, ShuffleSplit
from core_model import run_experiment
from data_processing import load_data, get_valid_indices_from_features, get_valid_indices_from_target, create_feature_matrix, preprocess_target, split_method_init, split_method_params
from other_func import process_dim, save_result

# ---- CONFIG ----
MODEL = 'lr'
SplitMethod = 'kf' # 'ss' represents shufflesplit, 'kf' represents KFold
MISSING_RATIOS = np.round(np.arange(0.1, 0.15, 0.05), decimals=2)
K_Fold = [10]

target = 'van_der_waals_radius'
EMBEDDING_COLUMNS = [
        'Mechanical properties',                # MECH  1
        'Optical properties',                   # OPT   2
        'Electrical and Magnetic properties',   # EM    3
        'Thermal properties',                   # THERM 4
        'Chemical properties',                  # CHEM  5
        'Atomic and radiational features',      # ARF   6
        'Applications',                         # APPL  7
        'Abundance'                             # ABND  8
    ]
# To facilitate the invocation by the master control script (exp2.py), the setting of this parameter has been moved to the 'if __name__ == '__main__':' code block at the end of the file.
# If you need to run this script independently to test a single feature, please go directly to the end of the file modify the value of the variable 'default_target_feat' (1-8, 0 represents all).

property_path = 'data/768_full.csv'
FILE_PATH = 'data/embedding_category_final.csv'
OUTPUT_PATH = f'results/embedding_sets/{MODEL}/test'
dataset_name = 'embedding_sets'
# ---- END ----

def prepare_base_data(file_path, prop_path, emb_cols, target_col):

    print('--- STEP 1: Preparing Base Data ---')
    df = load_data(file_path)
    property_df = load_data(prop_path)
    if df is None or property_df is None:
        return None, None, None

    valid_indices_x = get_valid_indices_from_features(df, emb_cols)
    valid_indices_y = get_valid_indices_from_target(property_df, target_col)
    final_valid_indices = np.intersect1d(valid_indices_x, valid_indices_y)
    
    df_filtered = df.loc[final_valid_indices].reset_index(drop=True)
    property_df_filtered = property_df.loc[final_valid_indices].reset_index(drop=True)

    y = preprocess_target(property_df_filtered, target_col)
    
    print('Base data prepared!\n')
    return df_filtered, property_df_filtered, y

def run_embedding_set_experiment(target_feat_to_run, df_clean, property_df_clean, y_clean):
    print(f"\n{'='*20} Running Experiment for target_feat = {target_feat_to_run} {'='*20}")

    X, dim = create_feature_matrix(df_clean, EMBEDDING_COLUMNS, target_feat_to_run)
    y = y_clean

    print(' DATA all preprocessed!\n')

    if y is not None:

        DIM_N_COMPONENTS = process_dim(X.shape[1])

        params_to_iterate, param_name = split_method_params(SplitMethod, MISSING_RATIOS, K_Fold)
        
        for param_value in params_to_iterate:
            split_method = split_method_init(SplitMethod, param_value)

            results_list = []
            progress_bar = tqdm(DIM_N_COMPONENTS, desc=f"Param ({param_name}={param_value})")
            for n in progress_bar:
                
                fold_metrics_list = []

                for train_index, test_index in split_method.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
                    metrics_for_fold = run_experiment(MODEL, n, X_train, X_test, y_train, y_test)
                    
                    # Store the results of this fold
                    fold_metrics_list.append(metrics_for_fold)
                
                # Average the metrics 
                fold_df = pd.DataFrame(fold_metrics_list)
                avg_metrics = fold_df.mean().to_dict()
                
                # print(f"Avg Result: R²={avg_metrics['r2']:.4f}, MAE={avg_metrics['mae']:.4f}, MSE={avg_metrics['mse']:.4f}, RMSE={avg_metrics['rmse']:.4f}")
 
                results_list.append({
                    'k': n,
                    'r2_score': avg_metrics['r2'],
                    'rmse': avg_metrics['rmse']
                })
            
            save_result(dataset_name, results_list, MODEL, SplitMethod, target, OUTPUT_PATH, target_feat_to_run, param_value)
        
    else:
        print(f"Since the target '{target}' cannot be processed, the model training for this target is skipped.")

if __name__ == '__main__':
    df_c, prop_df_c, y_c = prepare_base_data(FILE_PATH, property_path, EMBEDDING_COLUMNS, target)
    # target_feat starts from 1 to 8, 0 represents all features
    default_target_feat = 1
    run_embedding_set_experiment(target_feat_to_run=default_target_feat)