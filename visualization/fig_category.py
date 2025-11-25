import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
try:
    from visualization.missing_ratio import set_style
except ModuleNotFoundError:
    from missing_ratio import set_style

dim_limit = 768
FILES_TO_PLOT = [
    {
        'filepath': 'results/embedding_sets/lr/test/embedding_sets_targetfeat_1_lr_Metrics_van_der_waals_radius_KFold_10.csv', 
        'label': 'MECH'
    },
    {
        'filepath': 'results/embedding_sets/lr/test/embedding_sets_targetfeat_2_lr_Metrics_van_der_waals_radius_KFold_10.csv', 
        'label': 'OPT'
    },
    {
        'filepath': 'results/embedding_sets/lr/test/embedding_sets_targetfeat_3_lr_Metrics_van_der_waals_radius_KFold_10.csv', 
        'label': 'EM'
    },
    {
        'filepath': 'results/embedding_sets/lr/test/embedding_sets_targetfeat_4_lr_Metrics_van_der_waals_radius_KFold_10.csv', 
        'label': 'THERM'
    },
    {
        'filepath': 'results/embedding_sets/lr/test/embedding_sets_targetfeat_5_lr_Metrics_van_der_waals_radius_KFold_10.csv', 
        'label': 'CHEM'
    },
    {
        'filepath': 'results/embedding_sets/lr/test/embedding_sets_targetfeat_6_lr_Metrics_van_der_waals_radius_KFold_10.csv', 
        'label': 'ARF'
    },
    {
        'filepath': 'results/embedding_sets/lr/test/embedding_sets_targetfeat_7_lr_Metrics_van_der_waals_radius_KFold_10.csv', 
        'label': 'APPL'
    },
    {
        'filepath': 'results/embedding_sets/lr/test/embedding_sets_targetfeat_8_lr_Metrics_van_der_waals_radius_KFold_10.csv', 
        'label': 'ABND'
    }
]

current_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(current_script_dir)
OUTPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'test_category.png')

def main():
    # output path
    set_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    num_files = len(FILES_TO_PLOT)
    markers = ['o', 's', 'v', '^', 'D', '<', 'p', '>']
    cmap = plt.cm.get_cmap('Blues')
    color_start = 0.3
    color_end = 1
    color_values = np.linspace(color_start, color_end, num_files)
        
    for i, file_info in enumerate(FILES_TO_PLOT):
        filepath = os.path.join(PROJECT_ROOT, file_info['filepath'])
        label = file_info['label']
        color = cmap(color_values[i])
        
        if not os.path.exists(filepath):
            print(f"Warning: File not found, skipping '{filepath}'")
            continue
            
        df = pd.read_csv(filepath)
        df = df.sort_values(by='k')

        # Select only the rows where the value in the 'k' column is less than 768
        df = df[df['k'] < dim_limit].reset_index(drop=True)

        # Check if any data remains after filtering
        if df.empty:
            print(f"No data with k < {dim_limit} found in '{filepath}', skipping plot.")
            continue

        print(f"Plotting data from '{filepath}' (k < {dim_limit})...")
        marker_style = markers[i % len(markers)] # Cycle through marker styles
                    
        ax.plot(df['k'], df['rmse'], 
                    marker=marker_style,
                    label=label,
                    color=color,
                    markersize=5)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('black')
    
    ax.tick_params(axis='x', colors='black', labelsize=12)
    ax.tick_params(axis='y', colors='black', labelsize=12)
    ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles[::-1], labels[::-1], fontsize=12)
    legend.get_frame().set_alpha(0)

    ax.set_xlabel('Number of Features (N)', fontsize=14)
    ax.set_ylabel('RMSE', fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=300)

    print(f"\nSaved as: {OUTPUT_IMAGE_PATH}")
    plt.show()

if __name__ == '__main__':
    main()