# Refuse to overwrite existing result files
"""``guard_outputs(paths, force)``: abort before writing if any output exists.

Result files here are expensive to regenerate and not all are bit-reproducible
(``run_cforest`` does not seed R). Scripts call this up front so a re-run never
silently clobbers a previous run; pass ``--force`` to allow the overwrite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def guard_outputs(paths: Iterable[str | Path], force: bool = False) -> None:
    existing = [Path(p) for p in paths if Path(p).exists()]
    if not existing or force:
        return
    print('Refusing to overwrite existing result files:')
    for path in existing:
        print(f'  {path}')
    print('Move or rename them first (see _archive/), or pass --force to overwrite.')
    raise SystemExit(1)
