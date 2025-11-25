import argparse

# Import modules for the 'global' workflow
import global_exp
import visualization.normalized_RMSE as normalized_rmse_visualizer

# Import modules for the 'category' workflow
import embedding_sets_exp as ese
import visualization.fig_category as fcv

def run_global_workflow():
    """
    Executes the 'global' experiment workflow:
    1. Runs the experiments from global_exp.py to generate results.
    2. Calls normalized_RMSE.py for visualization.
    """
    print("=========================================================")
    print("         STARTING: Global Experiment Workflow          ")
    print("=========================================================")
    
    # --- STEP 1: Run the global experiment ---
    print("\n>>> STEP 1 of 2: Generating results for global experiment...")
    global_exp.main()
    print("\n>>> Global experiment finished. Result CSV files have been saved.")
    
    # --- STEP 2: Run visualization ---
    print("\n>>> STEP 2 of 2: Running visualization for global results...")
    normalized_rmse_visualizer.main()
    print("\n>>> Visualization finished. Plot has been saved.")
    
    print("\n=========================================================")
    print("          COMPLETED: Global Experiment Workflow          ")
    print("=========================================================")

def run_category_workflow():
    """
    Executes the 'category' (embedding sets) experiment workflow:
    1. Prepares the base data once.
    2. Loops 8 times to generate results for each feature category.
    3. Calls fig_category.py for visualization.
    """
    print("=========================================================")
    print("       STARTING: Category Experiment Workflow        ")
    print("=========================================================")
    
    # --- STEP 1: Prepare base data (runs only once) ---
    print("\n>>> STEP 1 of 3: Preparing base data for all categories...")
    df_clean, property_df_clean, y_clean = ese.prepare_base_data(
        ese.FILE_PATH, ese.property_path, ese.EMBEDDING_COLUMNS, ese.target
    )

    if y_clean is None:
        print("ERROR: Base data preparation failed. Aborting category workflow.")
        return
        
    # --- STEP 2: Loop through 8 categories to generate results ---
    print("\n>>> STEP 2 of 3: Generating results for all 8 feature categories...")
    for feat_index in range(1, 9):
        ese.run_embedding_set_experiment(
            target_feat_to_run=feat_index,
            df_clean=df_clean,
            property_df_clean=property_df_clean,
            y_clean=y_clean
        )
    print("\n>>> All 8 experiment CSV files have been generated.")
    
    # --- STEP 3: Run visualization ---
    print("\n>>> STEP 3 of 3: Running visualization for category results...")
    fcv.main()
    print("\n>>> Visualization finished. Plot has been saved.")
    
    print("\n=========================================================")
    print("        COMPLETED: Category Experiment Workflow        ")
    print("=========================================================")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Run machine learning experiments for element property prediction.",
        formatter_class=argparse.RawTextHelpFormatter # Keeps help message formatting
    )

    parser.add_argument(
        'experiment',
        nargs='?',
        choices=['global', 'category', 'all'],
        default='all',
        help=(
            "Specify which experiment workflow to run:\n"
            " 'global'   - Run the experiment using the global feature set.\n"
            " 'category' - Run 8 experiments for each feature category.\n"
            " 'all'      - Run both 'global' and 'category' workflows (default)."
        )
    )

    args = parser.parse_args()

    if args.experiment == 'all':
        run_global_workflow()
        print("\n" + "#" * 70 + "\n") # Add a clear separator between workflows
        run_category_workflow()
    elif args.experiment == 'global':
        run_global_workflow()
    elif args.experiment == 'category':
        run_category_workflow()

    print("\nAll selected workflows are complete!")