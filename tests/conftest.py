import matplotlib

matplotlib.use("Agg")  # must run before any test imports pyplot, so plotting works headless in CI

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sdg_dataframe():
    """A small synthetic dataset shaped like the real sdg CSV (post load_dataset)."""
    rng = np.random.RandomState(42)
    n = 40

    target = np.array(["A", "B", "C", "D"] * (n // 4))
    rng.shuffle(target)

    return pd.DataFrame({
        "Timestamp": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
        "F1": rng.choice(["success", "warning", "error", "unknown"], size=n),
        "F4": rng.choice(["QA Engineer", "Nurse", "Pilot"], size=n),
        "F5": rng.normal(size=n),
        "F17": np.where(rng.rand(n) < 0.9, np.nan, rng.normal(scale=100, size=n)),
        "F22": rng.uniform(0, 50, size=n),
        "Target": target,
    })
