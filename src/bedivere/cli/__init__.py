"""Command-line entry points.

    python -m bedivere.backtest   replay a spec, archive the run
    python -m bedivere.live       run a spec against live bars
    python -m bedivere.data       import CSVs, inspect the store, check coverage
    python -m bedivere.run        query the archive, the lock, and stop a run

The `python -m` names above are thin shims; the implementations live here so
the four commands can share one spec-to-composition path and one set of
boundary conventions (result to stdout, everything else to stderr, exit
0/1/130). See `common.py` for both.
"""
