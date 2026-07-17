"""Put src package roots on sys.path so short imports keep working.

After the layout refactor, libraries live under::

    src/common/          -> hyperparameters, run_config, ...
    src/experiments/     -> exp1, exp2, exp3
    src/models/          -> gpt2, llama, qwen

Entry scripts call ``ensure_src_on_path(__file__)`` once before importing them.
"""

from __future__ import annotations

import pathlib
import sys


def _find_src(start: pathlib.Path) -> pathlib.Path:
    p = start.resolve()
    if p.is_file():
        p = p.parent
    for candidate in (p, *p.parents):
        if (
            (candidate / "common").is_dir()
            and (candidate / "experiments").is_dir()
            and (candidate / "models").is_dir()
        ):
            return candidate
    raise RuntimeError(f"Could not locate src/ from {start}")


def ensure_src_on_path(start: str | pathlib.Path) -> pathlib.Path:
    """Insert ``src``, ``common``, ``experiments``, and ``models`` on ``sys.path``."""
    src = _find_src(pathlib.Path(start))
    for path in (src, src / "common", src / "experiments", src / "models"):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
    return src
