import pytest

from app import storage


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    """Every test writes into its own temporary project directory, never ./workspace."""
    monkeypatch.setattr(storage, "PROJECTS_DIR", tmp_path)
    monkeypatch.setenv("FILM_HOUSE_AUTOSNAPSHOT_SECONDS", "120")
    return tmp_path
