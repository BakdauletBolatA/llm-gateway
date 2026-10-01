"""Where the migrations are found when the package is installed, not run from a checkout."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_gateway.db.migrate import find_project_root


def make_project(root: Path) -> Path:
    (root / "migrations").mkdir(parents=True)
    (root / "alembic.ini").write_text("[alembic]\n")
    return root


def test_an_installed_package_finds_the_project_in_the_working_directory(tmp_path: Path) -> None:
    """In the image the code lives in site-packages and the project in /app."""
    site_packages = tmp_path / "usr" / "lib" / "python3.12" / "site-packages" / "llm_gateway"
    site_packages.mkdir(parents=True)
    app = make_project(tmp_path / "app")
    assert find_project_root([site_packages / "migrate.py", app]) == app


def test_a_checkout_is_found_from_the_source_file(tmp_path: Path) -> None:
    checkout = make_project(tmp_path / "repo")
    source = checkout / "src" / "llm_gateway" / "db"
    source.mkdir(parents=True)
    assert find_project_root([source / "migrate.py", tmp_path]) == checkout


def test_a_missing_project_is_an_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"alembic\.ini"):
        find_project_root([tmp_path])
