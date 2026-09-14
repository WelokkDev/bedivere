"""Make the src-layout package importable without installation, so a fresh
clone can run `pytest` (and the example) before any `pip install -e .`."""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


@pytest.fixture(autouse=True, scope="session")
def no_real_dotenv(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """The CLIs load `./.env`, and pytest runs from the repo root — exactly where
    a real one holds real keys. Point the loader at a file that does not exist,
    so no test can pick those keys up and reach a live account with them."""
    from bedivere.cli import common

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(common, "DOTENV_FILE", tmp_path_factory.mktemp("no-dotenv") / ".env")
        yield
