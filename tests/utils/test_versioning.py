import subprocess

from src.utils import versioning


def test_get_git_sha_returns_stripped_decoded_output(monkeypatch):
    def fake_check_output(cmd):
        assert cmd == ["git", "rev-parse", "--short", "HEAD"]
        return b"abc1234\n"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    assert versioning.get_git_sha() == "abc1234"


def test_get_git_sha_against_real_repo():
    sha = versioning.get_git_sha()

    assert isinstance(sha, str)
    assert sha
    assert " " not in sha
    assert "\n" not in sha
