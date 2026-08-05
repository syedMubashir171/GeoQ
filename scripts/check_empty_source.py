"""Reject Python files that parse but contain no code.

An empty ``.py`` file is valid Python: it compiles to an empty module and
imports without complaint. Nothing fails until some other module tries to
import a name from it, which may be many minutes into a run and far from the
actual cause. A file truncated to only its docstring behaves identically.

This hook makes that failure visible at commit time instead. It is separate
from ``check-ast``, which verifies that a file parses -- an empty file parses
perfectly well.

Usage:
    python scripts/check_empty_source.py FILE [FILE ...]

Exit status is 1 if any file is empty of executable content, 0 otherwise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

#  Files whose whole purpose may be to exist. A package marker with nothing in
#  it is correct Python; a docstring-only conftest is a legitimate placeholder.
#  Exempting them here rather than in .pre-commit-config.yaml keeps one source
#  of truth, so the hook behaves identically whether pre-commit invokes it or a
#  developer runs it by hand.
EXEMPT_NAMES = frozenset({"__init__.py", "conftest.py", "__main__.py"})


def has_no_code(path: Path) -> bool:
    """Return whether ``path`` contains no statements beyond a docstring.

    Args:
        path: File to inspect.

    Returns:
        True if the module body is empty or holds only a docstring.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        # Parse failures are check-ast's responsibility; reporting them here
        # too would produce two errors for one problem.
        return False

    body = tree.body
    if not body:
        return True
    if len(body) == 1 and isinstance(body[0], ast.Expr):
        value = body[0].value
        return isinstance(value, ast.Constant) and isinstance(value.value, str)
    return False


def main(argv: list[str]) -> int:
    """Check each path given on the command line.

    Args:
        argv: File paths.

    Returns:
        Process exit status.
    """
    paths = [Path(a) for a in argv if Path(a).name not in EXEMPT_NAMES]
    offenders = [p for p in paths if has_no_code(p)]
    for path in offenders:
        size = path.stat().st_size if path.exists() else 0
        print(
            f"{path}: contains no code ({size} bytes). "
            f"An empty module imports successfully and fails only when "
            f"something needs a name from it. If this file is intentionally "
            f"a placeholder, add it to EXEMPT_NAMES in this script.",
            file=sys.stderr,
        )
    return 1 if offenders else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
