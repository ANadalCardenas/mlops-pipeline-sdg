"""
Builds and returns the scikit-learn preprocessing + model Pipeline.

Detects numeric and categorical columns from the input DataFrame (skipping the
target and the excluded columns below), then constructs a ColumnTransformer with:
  - Numeric branch: SimpleImputer(strategy="median") + StandardScaler
  - Categorical branch: SimpleImputer(strategy="most_frequent") + OneHotEncoder

Wraps the transformer in a Pipeline with LogisticRegression(max_iter=1000) as
the final estimator. Does NOT fit the pipeline: fitting happens in train.py.

Public API:
    TARGET, EXCLUDED_COLUMNS
    build_training_pipeline(df: pd.DataFrame) -> sklearn.pipeline.Pipeline
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Target"

# Columns that are never used as features (see the data study in docs/ for details)
EXCLUDED_COLUMNS = [
    "Timestamp",                        # leakage: the month almost determines the class
    "F17",                              # 90% missing values
    "F4", "F7", "F8", "F11", "F13",     # job titles: ~110 unique values each, no signal
]


def build_training_pipeline(df: pd.DataFrame) -> Pipeline:
    exclude = {TARGET, *EXCLUDED_COLUMNS}

    numeric_cols = [c for c in df.select_dtypes(include=["int64", "float64"]).columns if c not in exclude]
    categorical_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in exclude]

    # Scaling puts features with very different ranges (±1 vs ±300) on the same footing,
    # which helps LogisticRegression converge and keeps its regularisation fair
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical columns need two steps: fill missing values first, then encode
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    # Combine preprocessing and the model into a single callable pipeline
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
