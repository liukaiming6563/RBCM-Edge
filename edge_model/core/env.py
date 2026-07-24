"""Runtime environment helpers for local RBCM scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def append_workspace_local_packages(project_root: str | Path) -> None:
    """Append project-local packages without shadowing the conda environment.

    Some optional evaluation dependencies may be installed under `.local_pkgs`
    for local Windows runs. The path is appended, not prepended, so the conda
    environment keeps priority for CUDA-sensitive packages such as torch.
    """

    local_pkgs = Path(project_root).resolve() / ".local_pkgs"
    if not local_pkgs.exists():
        return
    local_pkgs_str = str(local_pkgs)
    if local_pkgs_str not in sys.path:
        sys.path.append(local_pkgs_str)
