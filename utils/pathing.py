from pathlib import Path
from typing import Optional
import sys
from functools import lru_cache

MARKERS = {'limit_states', 'utils', 'config', 'active_learning', '.git'}

@lru_cache(maxsize=1)
def find_repo_root(start: Optional[Path] = None) -> Path:
    """Walk up until we find a folder that looks like the repo root."""
    p = (start or Path.cwd()).resolve()
    for parent in [p] + list(p.parents):
        if any((parent / m).exists() for m in MARKERS):
            return parent
    return p.parent.parent

def setup_repo_path(start: Optional[Path] = None) -> Path:
    """Find root, add to sys.path (idempotent), return root Path."""
    root = find_repo_root(start)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
