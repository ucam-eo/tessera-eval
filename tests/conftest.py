"""Pytest-wide setup.

This must run before `typer` (or anything importing it, e.g. tessera_eval.cli)
gets imported anywhere in the test session, so it lives in conftest.py with
nothing above it -- pytest imports conftest.py before collecting test modules.

Typer's rich_utils reads a handful of env vars *once*, at import time, to
decide how to render CLI errors:

    FORCE_TERMINAL = True if getenv("GITHUB_ACTIONS") or ... else None
    MAX_WIDTH = int(getenv("TERMINAL_WIDTH")) if set else None (auto-detect)

GitHub Actions always sets GITHUB_ACTIONS=true, which forces Typer's Rich
console into "real terminal" mode even though the runner has no actual tty.
Two independent problems follow, and both were observed to break
tests/test_cli.py assertions like `"--field is required" in result.output`
on CI only (see the tessera-eval#2 CI run for the original failure):

1. With a forced terminal, Rich syntax-highlights option names (e.g. bold
   "--field"), splitting the plain phrase with embedded ANSI codes even at
   a generous width -- `_TYPER_FORCE_DISABLE_TERMINAL=1` is Typer's own
   documented override to fully disable this (NO_COLOR alone doesn't cover
   it: bold/style codes aren't "color").
2. Whatever width Rich auto-detects for a forced-but-not-real terminal is
   environment-dependent (confirmed to differ between a local run, a Linux
   container, and the actual GitHub-hosted runner) and can be narrow enough
   to word-wrap the error message across box lines, breaking a contiguous
   substring match even with no ANSI codes at all. Pinning TERMINAL_WIDTH
   removes the auto-detection from the equation entirely.

Together these make the boxed CLI error panel deterministic: plain text,
wide enough to never wrap, everywhere this test suite runs.
"""

import os

os.environ.setdefault("_TYPER_FORCE_DISABLE_TERMINAL", "1")
os.environ.setdefault("TERMINAL_WIDTH", "200")
