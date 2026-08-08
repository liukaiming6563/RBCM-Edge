"""Run the lightweight public-release reproduction checks and generators."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(relative: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / relative), *args]
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "protocol", "tables", "figures", "all"),
        default="all",
    )
    parser.add_argument("--checkpoint-root", default="pretrained")
    args = parser.parse_args()

    stages = {args.stage} if args.stage != "all" else {"smoke", "protocol", "tables", "figures"}
    if "smoke" in stages:
        run(
            "scripts/release/smoke_paper_release.py",
            "--checkpoint-root",
            args.checkpoint_root,
            "--dataset",
            "all",
        )
    if "protocol" in stages:
        run(
            "scripts/checks/audit_multicue_strict_protocol.py",
            "--config",
            "edge_model/configs/rbcm/multicue_strict.yaml",
        )
        run(
            "scripts/checks/audit_nyud_strict_protocol.py",
            "--config",
            "edge_model/configs/rbcm/nyudv2_strict.yaml",
        )
    if "tables" in stages:
        run("scripts/analysis/build_formal_result_index.py")
        run("scripts/analysis/build_strict_protocol_tables.py")
        run("scripts/analysis/build_requested_cross_domain_report.py")
        run("scripts/analysis/build_v5_result_tables.py")
    if "figures" in stages:
        run("scripts/figures/edge/plot_joint_ablation_metrics.py")


if __name__ == "__main__":
    main()
