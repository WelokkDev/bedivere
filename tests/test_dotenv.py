"""`./.env` loading at the CLI boundary — the subset it reads, and the one rule
that matters most: a variable the environment already holds is never replaced."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bedivere.cli import common
from bedivere.cli.common import load_dotenv, run_command


def _load(tmp_path: Path, text: str, **existing: str) -> dict[str, str]:
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    environ = dict(existing)
    load_dotenv(path, environ=environ)
    return environ


def test_the_common_subset_is_read(tmp_path: Path) -> None:
    environ = _load(
        tmp_path,
        "# a comment\n"
        "\n"
        "PLAIN=db-abc123\n"
        "export EXPORTED=yes\n"
        "  SPACED  =  padded  \n"
        'DOUBLE="has # and spaces"\n'
        "SINGLE='x=y' # trailing comment\n"
        "INLINE=value # comment\n"
        "HASH_NO_SPACE=abc#def\n"
        "EMPTY=\n",
    )
    assert environ == {
        "PLAIN": "db-abc123",
        "EXPORTED": "yes",
        "SPACED": "padded",
        "DOUBLE": "has # and spaces",
        "SINGLE": "x=y",
        "INLINE": "value",
        "HASH_NO_SPACE": "abc#def",
        "EMPTY": "",
    }


def test_the_environment_wins_over_the_file(tmp_path: Path) -> None:
    # `KEY=... bedivere-data cost` has to override the file for one command.
    environ = _load(tmp_path, "DATABENTO_API_KEY=from-file\n", DATABENTO_API_KEY="from-shell")
    assert environ["DATABENTO_API_KEY"] == "from-shell"


def test_a_missing_file_is_not_an_error(tmp_path: Path) -> None:
    environ: dict[str, str] = {}
    load_dotenv(tmp_path / "absent.env", environ=environ)
    assert environ == {}


def test_a_bom_does_not_corrupt_the_first_key(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_bytes("﻿DATABENTO_API_KEY=db-key\n".encode())
    environ: dict[str, str] = {}
    load_dotenv(path, environ=environ)
    assert environ == {"DATABENTO_API_KEY": "db-key"}


def test_a_malformed_line_is_skipped_by_number_without_echoing_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    environ = _load(tmp_path, "GOOD=1\nthis-is-a-secret-pasted-alone\n1BAD=2\nALSO_GOOD=3\n")
    assert environ == {"GOOD": "1", "ALSO_GOOD": "3"}
    err = capsys.readouterr().err
    assert ":2 is not KEY=VALUE" in err and ":3 is not KEY=VALUE" in err
    assert "secret" not in err


def test_every_command_loads_it_before_its_body_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "BEDIVERE_TEST_DOTENV_VALUE"
    # load_dotenv writes os.environ behind monkeypatch's back; setting then
    # deleting through it first is what makes teardown remove the variable.
    monkeypatch.setenv(name, "")
    monkeypatch.delenv(name)
    (tmp_path / ".env").write_text(f"{name}=seen\n", encoding="utf-8")
    monkeypatch.setattr(common, "DOTENV_FILE", tmp_path / ".env")

    seen: list[str | None] = []

    def body() -> int:
        seen.append(os.environ.get(name))
        return 0

    assert run_command(body) == 0
    assert seen == ["seen"]
