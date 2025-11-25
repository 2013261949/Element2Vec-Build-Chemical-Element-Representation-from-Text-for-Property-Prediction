import numpy as np
import pandas as pd
import ast
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, ShuffleSplit

def load_data(file_path):
    print(f"Data is being loaded from '{file_path}'")
    try:
        df = pd.read_csv(file_path)
        print(" Successfully loaded.")
        return df
    except FileNotFoundError:
        print(f" Error: File not found at '{file_path}'. Please check if the file path is correct.")
        return None

def parse_vector_string(vector_str):

    if not isinstance(vector_str, str) or not vector_str.startswith('['):
        return np.array([])
    try:
        # ast.literal_eval
        return np.array(ast.literal_eval(vector_str))
    except (ValueError, SyntaxError) as e:
        print(f"ERROR : {e}. Return empty array")
        return np.array([])
    
def preprocess_target(df, target_column):
    print(f"\n--- Step 3: Preprocessing data for target '{target_column}' (y) ---")
    try:

        y_raw = df[[target_column]]
        
        if y_raw.isnull().sum().iloc[0] > 0:
            print(f"Found missing value in '{target_column}', padding with average value...")
            imputer = SimpleImputer(strategy='mean')
            y = imputer.fit_transform(y_raw)
        else:
            print("NO missing value. ")
            y = y_raw.values

        y = y.ravel()
        print(f"Target property data '{target_column}' has been ready with shape{y.shape}...\n")
        return y
    except KeyError:
        print(f"ERROR : Target property '{target_column}' doesn't exist ! ")
        return None
    except Exception as e:
        print(f"ERROR in processing : {e}")
        return None

def get_valid_indices_from_features(df, embedding_columns):
    """
    Scans the DataFrame to find indices of rows with valid feature embeddings.
    A row is valid if its embeddings can be parsed and the concatenated vector
    has a consistent dimension.

    Args:
        df (pd.DataFrame): The DataFrame containing feature data.
        embedding_columns (list): A list of feature column names to parse.

    Returns:
        list: A list of integer indices for the rows that are valid.
    """
    print("\n--- Step 2a: Validating features and finding valid indices (X) ---")
    
    valid_indices = []
    
    expected_dim = len(embedding_columns) * 3072

    print(f'expected dim is {expected_dim}')
    # Iterate through each row to check for validity
    for index, row in df.iterrows():
        try:
            row_vectors = [parse_vector_string(row[col]) for col in embedding_columns]
            
            # Check if any vector failed to parse (returned an empty array)
            if any(v.size == 0 for v in row_vectors):
                raise ValueError("One or more embedding vectors are empty or failed to parse.")

            concatenated_vector = np.concatenate(row_vectors)

            if expected_dim is None:
                expected_dim = len(concatenated_vector)
                print(f"Determined the expected feature dimension to be: {expected_dim}")

            # Check if the dimension of the current row is consistent
            if len(concatenated_vector) != expected_dim:
                raise ValueError(f"Dimension mismatch: Got {len(concatenated_vector)}, expected {expected_dim}.")

            # If all checks pass, this row's index is valid
            valid_indices.append(index)

        except Exception as e:
            # If the current row fails validation, skip it
            material_name = row.get('material', f'index {index}')
            # print(f" Skipping element '{material_name}' (index {index}), Reason: {e}")
            pass

    print(f"\nValidation complete. Found {len(valid_indices)} rows with valid features")
    return valid_indices

def get_valid_indices_from_target(df, target_column):
    """
    Finds the indices of rows that have a valid value in the specified target column.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the target column.
        target_column (str): The name of the target column.

    Returns:
        pd.Index: An index object containing the labels of the valid rows.
    """
    valid_indices = df.dropna(subset=[target_column]).index
    return valid_indices

'''
    need to modified the row_vectors setting before experiment
'''
def create_feature_matrix(df_clean, embedding_columns, target_feat):
    """
    Parses and concatenates embeddings from a pre-cleaned DataFrame to create the final feature matrix X.
    This function assumes all rows in df_clean are valid.

    Args:
        df_clean (pd.DataFrame): A DataFrame where all rows are guaranteed to have valid embeddings.
        embedding_columns (list): A list of feature column names to parse.

    Returns:
        tuple: (np.ndarray, int) containing the feature matrix X and its dimension, or (None, 0) if empty.
    """
    print("\n--- Step 2b: Creating final feature matrix (X) from clean data ---")
    
    processed_rows = []
    
    if df_clean.empty:
        print("ERROR: Input DataFrame is empty, cannot create feature matrix.")
        return None, 0

    # Iterate through the pre-filtered, clean DataFrame
    for index, row in df_clean.iterrows():
        # The try/except block is less critical here but kept for safety
        try:
            if target_feat > 0:
                row_vectors = [parse_vector_string(row.iloc[target_feat])]                      # single feature chosen
            else:                     
                row_vectors = [parse_vector_string(row[col]) for col in embedding_columns]      # all features in EMBEDDING_COLUMNS
            concatenated_vector = np.concatenate(row_vectors)
            processed_rows.append(concatenated_vector)
        except Exception as e:
            # This should ideally not happen if the validation function works correctly
            print(f"ERROR: Failed to process a supposedly clean row at new index {index}: {e}")
            return None, 0

    # Convert the list of vectors into the final NumPy matrix
    X = np.array(processed_rows)
    dim = X.shape[1]
    
    print(f"Feature matrix X created successfully. Shape: {X.shape}")
    return X, dim

def process_data_for_global(FILE_PATH, target):
    df = pd.read_csv(FILE_PATH)
    df_cleaned = df.dropna(subset=[target]).reset_index(drop=True)

    x_df = pd.DataFrame([eval(i) for i in df_cleaned['1']])
    y_s = df_cleaned[target]

    x_np = x_df.to_numpy()
    y_np = y_s.to_numpy()

    return x_np, y_np

def split_method_params(methodname, missing_ratios, k_fold):
    if methodname == 'ss':
        params_to_iterate = missing_ratios
        param_name = 'ratio'
    elif methodname == 'kf':
        params_to_iterate = k_fold
        param_name = 'k'
    else:
        raise ValueError(f"Unknown SplitMethod: {methodname}")
    
    return params_to_iterate, param_name

def split_method_init(methodname, param_value):
    if methodname == 'ss':
        print(f"\n--- Running with test ratio {param_value*100:.0f}% ---")
        split_method = ShuffleSplit(n_splits=5, test_size=param_value, random_state=54188)
    
    elif methodname == 'kf':
        print(f"\n--- Running for {param_value}-Fold Cross-Validation ---")
        split_method = KFold(n_splits=param_value, shuffle=True, random_state=54188)

    return split_method