"""Run the required paper-facing data and configuration preflight."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("edge_model/configs/rbcm/nyudv2_strict.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.check_data:
        raise SystemExit("This preflight currently requires --check-data.")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "checks" / "audit_nyud_strict_protocol.py"),
        "--config",
        str(args.config),
    ]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
