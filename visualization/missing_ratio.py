import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILE_PATTERN = r"analysis_results_(?:\w+_)?(\d+(\.\d+)?)_.*\.csv"
OUTPUT_FILENAME = "missing_ratio.png"
TARGET_DIRECTORIES = [
        'default',
        'default',
        'default'
    ]
def set_style():

    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Arial"],
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "figure.autolayout": True
    })

def process_all_directories(dir_list):

    all_results = {}
    for directory in dir_list:
        if os.path.isdir(directory):
            processed_df = process_files_in_directory_min(directory)
            if processed_df is not None:
                all_results[directory] = processed_df
        else:
            print(f"Warning: Directory '{directory}' not found. Skipping.")
    return all_results

def process_files_in_directory_avg(directory="."):

    plot_data = []
    
    print("Scanning for result files...")
    for filename in os.listdir(directory):
        match = re.match(FILE_PATTERN, filename)
        if match:
            try:
                n_splits = float(match.group(1))
                print(f"  Processing file: {filename} (n_splits={n_splits})")

                if n_splits >= 2:
                    test_ratio_percent = 1 / n_splits * 100
                else:
                    test_ratio_percent = n_splits * 100

                print(f'test ratio: {test_ratio_percent}')

                df = pd.read_csv(os.path.join(directory, filename))
                
                # Find the minimum RMSE and its corresponding k
                min_rmse_idx = df['rmse'].idxmin()
                k_at_min_rmse = df.loc[min_rmse_idx, 'k']
                
                # Filter the DataFrame for rows from that k value onwards
                df_tail = df[df['k'] >= k_at_min_rmse]
                
                # Calculate the average RMSE of this "best tail"
                avg_tail_rmse = df_tail['rmse'].mean()

                plot_data.append({
                    'train_ratio_percent': test_ratio_percent,
                    'avg_tail_rmse': avg_tail_rmse
                })

            except (ValueError, FileNotFoundError, pd.errors.EmptyDataError) as e:
                print(f"  Could not process file {filename}. Reason: {e}")

    if not plot_data:
        print("\nError: No valid data was processed. Exiting.")
        return None

    results_df = pd.DataFrame(plot_data).sort_values(by='train_ratio_percent')
    output_csv_path = 'summary_results.csv'

    results_df.to_csv(output_csv_path, index=False, float_format='%.4f')

    print(f"\n✅ Summary results successfully saved to: {output_csv_path}")
    return results_df

def process_files_in_directory_min(directory="."):

    plot_data = []
    
    print("Scanning for result files...")
    for filename in os.listdir(directory):
        match = re.match(FILE_PATTERN, filename)
        if match:
            try:
                n_splits = float(match.group(1))

                print(f"  Processing file: {filename} (n_splits={n_splits})")

                test_ratio_percent = n_splits
                if n_splits >= 2:
                    test_ratio_percent = 1 / n_splits * 100
                else:
                    test_ratio_percent = n_splits * 100
                print(f'  test ratio: {test_ratio_percent}')

                df = pd.read_csv(os.path.join(directory, filename))
                
                min_rmse_value = df['rmse'].min()

                plot_data.append({
                    'train_ratio_percent': test_ratio_percent,
                    'avg_tail_rmse': min_rmse_value 
                })

            except (ValueError, FileNotFoundError, pd.errors.EmptyDataError) as e:
                print(f"  Could not process file {filename}. Reason: {e}")

    if not plot_data:
        print("\nError: No valid data was processed. Exiting.")
        return None
        
    results_df = pd.DataFrame(plot_data).sort_values(by='train_ratio_percent')

    output_csv_path = f'summary_results_{directory}.csv'
    results_df.to_csv(output_csv_path, index=False, float_format='%.4f')
    print(f"\n✅ Summary results successfully saved to: {output_csv_path}")
    
    return results_df

def create_plot(df):

    print("\nCreating plot...")
    set_style()
    colors = ['#1a70de', '#f24040', '#525252']
    markers = ['^', 'o', 's']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (dir_name, df) in enumerate(df.items()):
        legend_label = f"{dir_name.replace('_', ' ')}"
        ax.plot(
            df['train_ratio_percent'],
            df['avg_tail_rmse'],
            marker=markers[i % len(markers)],
            markersize=8,
            linestyle='-',
            linewidth=2.5,
            color=colors[i % len(colors)],
            label=legend_label
        )
    ax.legend(
        loc='lower right', 
        fontsize='large',
        frameon=True,
        shadow=True,
        edgecolor='black'
    )

    # Labels
    ax.set_xlabel("Missing Ratio (%)", fontweight='bold')
    ax.set_ylabel("Best Tail Average RMSE (Å)", fontweight='bold')

    # Ticks formatting
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    plt.savefig(OUTPUT_FILENAME, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {OUTPUT_FILENAME}")
    plt.show()

def main():
    results_df = process_all_directories(TARGET_DIRECTORIES)

    if results_df is not None:
        create_plot(results_df)

if __name__ == "__main__":
    main()