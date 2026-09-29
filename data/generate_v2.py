import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = Path(__file__).resolve().parent

V1_PATH = DATA_DIR / "v1" / "sdg.csv"
V2_DIR = DATA_DIR / "v2"

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# -----------------------------
# Utility
# -----------------------------
def ensure_dirs():
    V2_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# v2: More data & Drifted distributions
# -----------------------------
def create_v2(df: pd.DataFrame):
    print("Generating v2...")

    # CHANGE 0: Increase volume by 40%
    # Resampled rows get a little gaussian noise on the dense numeric columns
    # so they are not exact duplicates of v1 rows
    print("Increased volume...")
    additional_rows = df.sample(frac=0.4, replace=False, random_state=RANDOM_SEED).copy()
    dense_cols = ["F5", "F10", "F15", "F20", "F22", "F23", "F25", "F26", "F28", "F29", "F30"]
    for col in dense_cols:
        noise = rng.normal(scale=0.05 * df[col].std(), size=len(additional_rows))
        additional_rows[col] = additional_rows[col] + noise
    v2_df = pd.concat([df, additional_rows], ignore_index=True)

    print("Adding distribution drift...")

    # CHANGE 1: Class distribution drift, the minority class D becomes more frequent
    print("Applying Target distribution drift...")
    extra_d = v2_df[v2_df["Target"] == "D"].sample(frac=2.0, replace=True, random_state=RANDOM_SEED).copy()
    extra_d["F22"] = extra_d["F22"] + rng.normal(scale=1.0, size=len(extra_d))
    v2_df = pd.concat([v2_df, extra_d], ignore_index=True)

    # CHANGE 2: F22 drift, a small offset plus extra noise (e.g. sensor recalibration)
    print("Applying F22 drift...")
    v2_df["F22"] = v2_df["F22"] + 1.5 + rng.normal(scale=2.0, size=len(v2_df))

    # CHANGE 3: Status drift, more "error" values in F1 (upstream system degradation)
    print("Applying F1 status drift...")
    flip = rng.random(len(v2_df)) < 0.15
    v2_df.loc[flip & v2_df["F1"].notna(), "F1"] = "error"

    # Shuffle so the appended rows are not all at the end
    v2_df = v2_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    v2_df.to_csv(V2_DIR / "sdg.csv", index=False)

# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dirs()

    print("Loading v1 dataset...")
    df_v1 = pd.read_csv(V1_PATH)

    create_v2(df_v1)

    print("\nData generation completed:")
    print(" - data/v2/sdg.csv")

if __name__ == "__main__":
    main()
