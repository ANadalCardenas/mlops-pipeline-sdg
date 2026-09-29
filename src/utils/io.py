"""
Thin wrappers around JSON file I/O using the standard library `json` module.

Public API:
    load_json(path: str) -> dict       — reads and parses a JSON file
    save_json(data: dict, path: str)   — serialises a dict and writes it to disk
"""

import json


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    with open(path, "w") as f:
        # indent=2 makes the file human-readable when inspected manually
        json.dump(data, f, indent=2)
