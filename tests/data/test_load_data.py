import pandas as pd
import pytest

from src.data import load_data


def test_load_dataset_keeps_string_labels_and_drops_unlabelled_rows(tmp_path, monkeypatch):
    csv_path = tmp_path / "sdg.csv"
    pd.DataFrame({
        "F1": ["success", "error", "warning"],
        "F22": [1.0, 2.0, 3.0],
        "Target": ["A", None, "D"],
    }).to_csv(csv_path, index=False)

    monkeypatch.setitem(load_data.DATA_PATHS, "test", str(csv_path))

    df = load_data.load_dataset("test")

    assert df["Target"].tolist() == ["A", "D"]
    assert df.index.tolist() == [0, 1]


def test_load_dataset_unknown_version_raises_key_error():
    with pytest.raises(KeyError):
        load_data.load_dataset("does-not-exist")
