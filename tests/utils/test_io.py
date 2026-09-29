import json

import pytest

from src.utils.io import load_json, save_json


def test_save_json_round_trip(tmp_path):
    data = {"accuracy": 0.9, "labels": ["a", "b"], "nested": {"x": 1}}
    path = tmp_path / "metrics.json"

    save_json(data, str(path))

    assert path.exists()
    assert load_json(str(path)) == data


def test_save_json_is_indented(tmp_path):
    path = tmp_path / "metrics.json"

    save_json({"a": 1}, str(path))

    content = path.read_text()
    assert "\n" in content  # indent=2 produces multi-line output
    assert json.loads(content) == {"a": 1}


def test_load_json_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError):
        load_json(str(missing))
