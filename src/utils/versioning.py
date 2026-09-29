"""
Provides the current Git commit SHA to be stamped into model_info.json,
enabling full traceability between a saved model artifact and the exact code
revision that produced it.

Public API:
    get_git_sha() -> str   — returns the short commit SHA via subprocess
"""

import subprocess


def get_git_sha() -> str:
    # --short gives an 8-character hash that is compact but still unique enough
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
