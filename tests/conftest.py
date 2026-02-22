from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def tmp_path() -> Path:
    """
    Workspace-local tmp_path replacement for environments where OS temp
    directories are not writable during test execution.
    """
    base = Path.cwd() / ".test_tmp"
    base.mkdir(parents=True, exist_ok=True)

    path = base / f"case_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)

    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
