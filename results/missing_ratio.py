import os
import re
import argparse
import pandas as pd
import numpy as np

# === you only need to change this line when switching property ===
DEFAULT_BASE_DIR = r"youngs_modulus"
# e.g. DEFAULT_BASE_DIR = r"boiling_point", r"van_der_waals_radius", ...

# Match files like:
# embedding_sets_targetfeat_1_lr_Metrics_van_der_waals_radius_ShuffleSplit_Ratio_0.05_repeat5.csv
FILE_PATTERN = re.compile(
    r".*ShuffleSplit_Ratio_(\d+(?:\.\d+)?)_repeat5\.csv$",
    re.IGNORECASE
)

def compute_tail_avg_rmse(csv_path: str) -> float:
    """
    For a given CSV:
    - Find the row with the minimum RMSE.
    - Take its k as k*.
    - Return the average RMSE over all rows with k >= k*.
    """
    df = pd.read_csv(csv_path)
    for need in ("k", "rmse"):
        if need not in df.columns:
            raise ValueError(f"{os.path.basename(csv_path)} is missing required column '{need}'")
    idx_min = df["rmse"].idxmin()
    k_star = df.loc[idx_min, "k"]
    return float(df[df["k"] >= k_star]["rmse"].mean())

def build_table_for_lr(base_dir: str) -> pd.DataFrame:
    """
    Scan base_dir for files that match FILE_PATTERN, compute the
    tail-average RMSE for each missing-ratio, and return a 2-column table:
    - 'Missing Ratio (%)'
    - 'rmse'
    """
    expected_ratios = [int(round(x * 100)) for x in np.arange(0.05, 1.0, 0.05)]
    values = {}

    for fn in os.listdir(base_dir):
        m = FILE_PATTERN.match(fn)
        if not m:
            continue
        ratio_float = float(m.group(1))
        ratio_percent = int(round(ratio_float * 100))
        csv_path = os.path.join(base_dir, fn)
        try:
            val = compute_tail_avg_rmse(csv_path)
            values[ratio_percent] = val
        except Exception as e:
            print(f"[WARN] Skipping {fn}: {e}")

    table = pd.DataFrame({
        "Missing Ratio (%)": expected_ratios,
        "rmse": [values.get(r, np.nan) for r in expected_ratios],
    })
    missing = [r for r in expected_ratios if r not in values]
    if missing:
        print(f"[INFO] These ratios are missing or failed to parse; filled with NaN: {missing}")
    return table

def main():
    ap = argparse.ArgumentParser(
        description="Aggregate embedding_sets LR results into a two-column missing-ratio table"
    )
    ap.add_argument(
        "--base_dir",
        default=DEFAULT_BASE_DIR,
        help=f"Directory containing ShuffleSplit_Ratio_*.csv files (default: {DEFAULT_BASE_DIR})"
    )
    ap.add_argument(
        "--out",
        default="missing_ratio_lr.csv",
        help="Output CSV filename"
    )
    args = ap.parse_args()

    if not os.path.isdir(args.base_dir):
        print(f"Directory does not exist: {args.base_dir}")
        return

    table = build_table_for_lr(args.base_dir)
    out_path = (
        os.path.join(args.base_dir, args.out)
        if not os.path.isabs(args.out)
        else args.out
    )
    table.to_csv(out_path, index=False, float_format="%.6f")
    print(f"✅ Saved missing-ratio table: {out_path}")

if __name__ == "__main__":
    main()
