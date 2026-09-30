from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "external" / "TinyRecursiveModels"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
