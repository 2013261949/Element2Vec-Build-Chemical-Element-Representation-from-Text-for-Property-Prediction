import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import tqdm
from other_func import save_result, process_dim
from core_model import run_experiment
from data_processing import process_data_for_global, split_method_params, split_method_init

# ---- CONFIG ----
MODEL = 'lr'
SplitMethod = 'kf' # 'ss' represents shufflesplit, 'kf' represents KFold
MISSING_RATIOS = np.round(np.arange(0.1, 0.15, 0.05), decimals=2)
K_Fold = [10]

targets = [
    'van_der_waals_radius',
    'youngs_modulus',
    'melting_point',
    'molar_volume',
    'rou']
target_feat = 0

FILE_PATH = 'data/768_full.csv'
OUTPUT_PATH = f'results/global/{MODEL}/test'
datasetname = 'global'
# ---- END ----

def main():

    for target in targets:
        x_np, y_np = process_data_for_global(FILE_PATH, target)
        DIM_N_COMPONENTS = process_dim(x_np.shape[1])

        params_to_iterate, param_name = split_method_params(SplitMethod, MISSING_RATIOS, K_Fold)

        for param_value in params_to_iterate:
            split_method = split_method_init(SplitMethod, param_value)

            results_list = []
            progress_bar = tqdm(DIM_N_COMPONENTS, desc=f"Param ({param_name}={param_value})")
            for n in progress_bar:
                
                fold_metrics_list = []

                for train_index, test_index in split_method.split(x_np):
                    x_train, x_test = x_np[train_index], x_np[test_index]
                    y_train, y_test = y_np[train_index], y_np[test_index]

                    metrics_for_fold = run_experiment(MODEL, n, x_train, x_test, y_train, y_test)
                    
                    fold_metrics_list.append(metrics_for_fold)
                
                fold_df = pd.DataFrame(fold_metrics_list)
                avg_metrics = fold_df.mean()
                results_list.append({
                    'k': n,
                    'r2_score': avg_metrics['r2'],
                    'rmse': avg_metrics['rmse'] 
                })
            save_result(datasetname, results_list, MODEL, SplitMethod, target, OUTPUT_PATH, target_feat, param_value)

if __name__ == '__main__':
    main()