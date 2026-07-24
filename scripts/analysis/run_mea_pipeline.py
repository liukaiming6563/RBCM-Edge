"""Run the final MEA analysis and figure modules in a reproducible order."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "MEA_outputs"


@dataclass(frozen=True)
class Step:
    name: str
    script: Path


STEPS = (
    Step(
        "final_ume_cme_analysis",
        ROOT / "MEA_analysis" / "run_MEA_final_UME_CME_trajectory_analysis.py",
    ),
    Step(
        "01_sampled_spike_raster",
        ROOT / "MEA_model" / "01_sampled_spike_raster" / "plot_01_sampled_spike_raster.py",
    ),
    Step(
        "02_unit_spatial_distribution",
        ROOT / "MEA_model" / "02_unit_spatial_distribution" / "plot_02_unit_spatial_distribution.py",
    ),
    Step(
        "03_local_population_trajectory_examples",
        ROOT
        / "MEA_model"
        / "03_local_population_trajectory_examples"
        / "plot_03_local_population_trajectory_examples.py",
    ),
    Step(
        "04_grid_trajectory_mae_maps",
        ROOT
        / "MEA_model"
        / "04_grid_trajectory_mae_maps"
        / "plot_04_grid_trajectory_mae_maps.py",
    ),
    Step(
        "04_grid_trajectory_mae_maps_no_legend_text",
        ROOT
        / "MEA_model"
        / "04_grid_trajectory_mae_maps_no_legend_text"
        / "plot_04_grid_trajectory_mae_maps_no_legend_text.py",
    ),
    Step(
        "05_trajectory_mae_permutation_test_summary",
        ROOT
        / "MEA_model"
        / "05_trajectory_mae_permutation_test_summary"
        / "plot_05_trajectory_mae_permutation_test_summary.py",
    ),
    Step(
        "06_shape_mae_grid_maps",
        ROOT / "MEA_model" / "06_shape_mae_grid_maps" / "plot_06_shape_mae_grid_maps.py",
    ),
    Step(
        "07_shape_mae_permutation_test_summary",
        ROOT
        / "MEA_model"
        / "07_shape_mae_permutation_test_summary"
        / "plot_07_shape_mae_permutation_test_summary.py",
    ),
    Step(
        "08_derivative_mae_grid_maps",
        ROOT
        / "MEA_model"
        / "08_derivative_mae_grid_maps"
        / "plot_08_derivative_mae_grid_maps.py",
    ),
    Step(
        "09_derivative_mae_permutation_test_summary",
        ROOT
        / "MEA_model"
        / "09_derivative_mae_permutation_test_summary"
        / "plot_09_derivative_mae_permutation_test_summary.py",
    ),
    Step(
        "10_directional_stability_summary",
        ROOT
        / "MEA_model"
        / "10_directional_stability_summary"
        / "plot_10_directional_stability_summary.py",
    ),
    Step(
        "11_grid_scale_stability_summary",
        ROOT
        / "MEA_model"
        / "11_grid_scale_stability_summary"
        / "plot_11_grid_scale_stability_summary.py",
    ),
    Step(
        "12_effect_distribution_summary",
        ROOT
        / "MEA_model"
        / "12_effect_distribution_summary"
        / "plot_12_effect_distribution_summary.py",
    ),
    Step(
        "13_permutation_null_examples",
        ROOT
        / "MEA_model"
        / "13_permutation_null_examples"
        / "plot_13_permutation_null_examples.py",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--from-step", choices=[step.name for step in STEPS])
    parser.add_argument("--to-step", choices=[step.name for step in STEPS])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_steps(args: argparse.Namespace) -> tuple[Step, ...]:
    names = [step.name for step in STEPS]
    start = names.index(args.from_step) if args.from_step else 0
    stop = names.index(args.to_step) + 1 if args.to_step else len(STEPS)
    if start >= stop:
        raise ValueError("--from-step must not occur after --to-step")
    return STEPS[start:stop]


def main() -> None:
    args = parse_args()
    if args.list:
        for index, step in enumerate(STEPS, start=1):
            print(f"{index:02d} {step.name}: {step.script.relative_to(ROOT)}")
        return

    logs = OUTPUT_ROOT / "launcher_logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["RBCM_EDGE_PROJECT_ROOT"] = str(ROOT)

    for step in selected_steps(args):
        if not step.script.is_file():
            raise FileNotFoundError(step.script)
        command = [args.python, str(step.script)]
        print(f"[MEA] {step.name}")
        print("+ " + " ".join(command))
        if args.dry_run:
            continue
        with (logs / f"{step.name}.log").open(
            "w", encoding="utf-8", errors="replace"
        ) as handle:
            subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )


if __name__ == "__main__":
    main()
