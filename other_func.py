import os
import pandas as pd

def save_result(dataset_name, results_list, model_name, SplitName, target, path, target_feat, ratio_para):
    os.makedirs(path, exist_ok=True)
    results_df = pd.DataFrame(results_list)
    if SplitName == 'ss':
        output_filename = f'{dataset_name}_targetfeat_{target_feat}_{model_name}_Metrics_{target}_ShuffleSplit_Ratio_{ratio_para}_repeat5.csv'
    if SplitName == 'kf':
        output_filename = f'{dataset_name}_targetfeat_{target_feat}_{model_name}_Metrics_{target}_KFold_{ratio_para}.csv'
    
    full_output_path = os.path.join(path, output_filename)

    results_df.to_csv(full_output_path, index=False, float_format='%.4f')
    print(f"\nSaved as: {full_output_path}")

def process_dim(x_dim):

    n_values = [x_dim]
    current_val = int(x_dim // 1000) * 1000

    while current_val >= 1000:

        if current_val < n_values[-1]:
            n_values.append(current_val)
        current_val -= 100
        
    current_val = 900
    while current_val >= 100:
        if current_val < n_values[-1]:
             n_values.append(current_val)
        current_val -= 10   
    return n_values