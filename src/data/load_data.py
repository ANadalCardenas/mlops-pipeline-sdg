"""
Responsible for reading the versioned CSV dataset from disk and returning a
pandas DataFrame. Maps data version strings (e.g. "v1", "v2") to their
corresponding file paths and reads the CSV with pandas.

The target column `Target` holds the class labels ("A", "B", "C", "D") and is
returned as-is: sklearn handles string labels natively in multiclass problems.

Public API:
    load_dataset(data_version: str) -> pd.DataFrame
"""

import pandas as pd


# Map version strings to their CSV paths; extend here when new data versions arrive
DATA_PATHS = {
    "v1": "data/v1/sdg.csv",
    "v2": "data/v2/sdg.csv",
}


def load_dataset(data_version: str) -> pd.DataFrame:
    path = DATA_PATHS[data_version]
    df = pd.read_csv(path)

    # Rows without a label can't be used for supervised training
    return df.dropna(subset=["Target"]).reset_index(drop=True)
