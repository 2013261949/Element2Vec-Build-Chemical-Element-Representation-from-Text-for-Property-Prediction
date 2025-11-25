import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
try:
    from visualization.missing_ratio import set_style
except ModuleNotFoundError:
    from missing_ratio import set_style



FILES_TO_PLOT = [
    {
        'filepath': 'results/global/lr/test/global_targetfeat_0_lr_Metrics_van_der_waals_radius_KFold_10.csv', 
        'label': 'Van Der Waals Radius'
    },
    {
        'filepath': 'results/global/lr/test/global_targetfeat_0_lr_Metrics_molar_volume_KFold_10.csv', 
        'label': 'Molar Volume'
    },
    {
        'filepath': 'results/global/lr/test/global_targetfeat_0_lr_Metrics_melting_point_KFold_10.csv', 
        'label': 'Melting Temp'
    },
    {
        'filepath': 'results/global/lr/test/global_targetfeat_0_lr_Metrics_youngs_modulus_KFold_10.csv', 
        'label': 'Youngs Modulus'
    },
    {
        'filepath': 'results/global/lr/test/global_targetfeat_0_lr_Metrics_rou_KFold_10.csv', 
        'label': 'Density'
    }
]

current_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(current_script_dir)
OUTPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'test_global.png')

def main():
    num_files = len(FILES_TO_PLOT)
    # --- Plotting Setup ---
    set_style()
    fig, ax = plt.subplots(figsize=(10,6))

    cmap = plt.cm.get_cmap('CMRmap')
    color_start = 0.1
    color_end = 0.9
    color_values = np.linspace(color_start, color_end, num_files)

    markers = ['o', 's', 'v', '^', 'D', 'x']


    # --- Main Plotting Loop ---

    for i, file_info in enumerate(FILES_TO_PLOT):
        filepath = os.path.join(PROJECT_ROOT, file_info['filepath'])
        label = file_info['label']
        
        if not os.path.exists(filepath):
            print(f"Warning: File not found, skipping '{filepath}'")
            continue
            
        df = pd.read_csv(filepath)
        # Ensure data is sorted by 'k' for a smooth line plot
        df = df.sort_values(by='k').reset_index(drop=True)
        
        min_rmse_local = df['rmse'].min()
        max_rmse_local = df['rmse'].max()
        
        # Perform LOCAL normalization
        if (max_rmse_local - min_rmse_local) > 0:
            df['normalized_rmse'] = (df['rmse'] - min_rmse_local) / (max_rmse_local - min_rmse_local)
        else:
            print('error')
            quit()
        
        color = cmap(color_values[i])
        marker = markers[i % len(markers)]
        
        print(f"Plotting data from '{filepath}'...")
        
        # Plot the LOCALLY NORMALIZED RMSE
        ax.plot(df['k'], df['normalized_rmse'], marker=marker, linestyle='-', color=color, markersize=5, linewidth=2, label=label)

        # Find and annotate the best point (lowest original RMSE) for this curve
        best_idx = df['rmse'].idxmin()
        best_k = df.loc[best_idx, 'k']
        best_original_rmse = df.loc[best_idx, 'rmse']

        text_offset = (best_k + 25, 0.1) 
        # Manually adjust offset for specific labels
        if 'Density' in label:
            text_offset = (best_k + 30, 0.2) 
        elif 'Youngs' in label:
            text_offset = (best_k - 50, 0.32)
        elif 'Melting' in label:
            text_offset = (best_k - 120, 0.2)
        elif 'Molar' in label:
            text_offset = (best_k + 110, 0.32)
        elif 'Van' in label:
            text_offset = (best_k + 180, 0.2)
        ax.annotate(

            f"Best for '{label}':\n N = {best_k}",
            xy=(best_k, 0), # The annotation always points to y=0 for the best point
            xytext=text_offset, # Adjust text offset
            arrowprops=dict(arrowstyle="fancy", color=color, alpha=0.8, connectionstyle="arc3,rad=0.1"),
            fontsize=11,
            color='black',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='white', edgecolor=color, alpha=0.8, linewidth=2),
            ha='center',
            va='center'
        )

    # --- Final Plot Formatting ---
    ax.set_xlabel('Number of Top Features (N)', fontsize=14)
    ax.set_ylabel('Normalized RMSE', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(False)

    ax.set_ylim(-0.05, 1.15)

    plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to: {OUTPUT_IMAGE_PATH}")
    plt.show()

if __name__ == '__main__':
    main()