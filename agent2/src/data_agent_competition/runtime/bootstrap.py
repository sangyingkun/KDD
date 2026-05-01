from __future__ import annotations

import sys
from pathlib import Path


def ensure_import_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
