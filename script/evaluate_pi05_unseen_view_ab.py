#!/usr/bin/env python3
"""Compare frozen pickplace PI0.5 with and without Adapter on unseen views."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "script" / "evaluate_pi05_unseen_camera_views.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-policy-path", type=Path, required=True)
    parser.add_argument("--adapter-policy-path", type=Path, required=True)
    parser.add_argument("--server-address", default="127.0.0.1:8081")
    parser.add_argument("--episodes-per-view", type=int, default=10)
    parser.add_argument("--scenario-seeds-file", type=Path, required=True)
    parser.add_argument("--policy-inference-seed", type=int, default=0)
    parser.add_argument("--suite", action="append")
    parser.add_argument("--view", action="append")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "unseen_camera_view_ab",
    )
    parser.add_argument("--run-name")
    return parser.parse_args()


def run_evaluator(
    cli: argparse.Namespace,
    policy_path: Path,
    run_dir: Path,
    adapter_enabled: bool,
) -> None:
    command = [
        sys.executable,
        str(EVALUATOR),
        "--policy-path",
        str(policy_path),
        "--server-address",
        cli.server_address,
        "--episodes-per-view",
        str(cli.episodes_per_view),
        "--scenario-seeds-file",
        str(cli.scenario_seeds_file.expanduser().resolve()),
        "--policy-inference-seed",
        str(cli.policy_inference_seed),
        "--run-dir",
        str(run_dir),
    ]
    command.append(
        "--feature-adapter-enabled" if adapter_enabled else "--no-feature-adapter-enabled"
    )
    for suite in cli.suite or []:
        command.extend(["--suite", suite])
    for view in cli.view or []:
        command.extend(["--view", view])
    if cli.max_steps is not None:
        command.extend(["--max-steps", str(cli.max_steps)])
    subprocess.run(command, cwd=ROOT, check=True)


def load_results(path: Path) -> dict:
    with path.open(encoding="utf-8") as result_file:
        return json.load(result_file)


def paired_counts(base_view: dict, adapter_view: dict) -> dict[str, int]:
    base_by_seed = {episode["seed"]: episode for episode in base_view["episode_results"]}
    adapter_by_seed = {
        episode["seed"]: episode for episode in adapter_view["episode_results"]
    }
    if len(base_by_seed) != len(base_view["episode_results"]):
        raise RuntimeError(f"Duplicate baseline seed in view {base_view['id']}")
    if len(adapter_by_seed) != len(adapter_view["episode_results"]):
        raise RuntimeError(f"Duplicate Adapter seed in view {adapter_view['id']}")
    if base_by_seed.keys() != adapter_by_seed.keys():
        raise RuntimeError(f"Base and Adapter used different seeds for view {base_view['id']}")

    counts = {
        "both_success": 0,
        "both_failure": 0,
        "improved": 0,
        "regressed": 0,
    }
    for seed, base_episode in base_by_seed.items():
        adapter_episode = adapter_by_seed[seed]
        base_success = bool(base_episode["success"])
        adapter_success = bool(adapter_episode["success"])
        if base_success and adapter_success:
            counts["both_success"] += 1
        elif not base_success and not adapter_success:
            counts["both_failure"] += 1
        elif not base_success and adapter_success:
            counts["improved"] += 1
        else:
            counts["regressed"] += 1
    counts["net_paired_improvement"] = counts["improved"] - counts["regressed"]
    return counts


def sum_paired_counts(rows: list[dict]) -> dict[str, int]:
    keys = (
        "both_success",
        "both_failure",
        "improved",
        "regressed",
        "net_paired_improvement",
    )
    return {key: sum(row[key] for row in rows) for key in keys}


def write_comparison(run_dir: Path, base_results: dict, adapter_results: dict) -> None:
    base_by_view = {view["id"]: view for view in base_results["views"]}
    adapter_by_view = {view["id"]: view for view in adapter_results["views"]}
    if base_by_view.keys() != adapter_by_view.keys():
        raise RuntimeError("Base and Adapter evaluated different unseen view sets")
    if base_results["shared_valid_seeds"] != adapter_results["shared_valid_seeds"]:
        raise RuntimeError("Base and Adapter used different scenario seeds")
    if base_results["policy_inference_seed"] != adapter_results["policy_inference_seed"]:
        raise RuntimeError("Base and Adapter used different policy inference seeds")

    rows = []
    for view_id, base_view in base_by_view.items():
        adapter_view = adapter_by_view[view_id]
        paired = paired_counts(base_view, adapter_view)
        rows.append(
            {
                "view": view_id,
                "suite": base_view["suite"],
                "description": base_view["description"],
                "episodes": base_view["episodes"],
                "base_successes": base_view["successes"],
                "base_rate": base_view["success_rate"],
                "adapter_successes": adapter_view["successes"],
                "adapter_rate": adapter_view["success_rate"],
                "delta_rate": adapter_view["success_rate"] - base_view["success_rate"],
                **paired,
            }
        )

    with (run_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    suite_summary = {}
    for suite in sorted({row["suite"] for row in rows}):
        suite_rows = [row for row in rows if row["suite"] == suite]
        suite_summary[suite] = {
            "views": len(suite_rows),
            "episodes": sum(row["episodes"] for row in suite_rows),
            "base_macro_rate": sum(row["base_rate"] for row in suite_rows) / len(suite_rows),
            "adapter_macro_rate": sum(row["adapter_rate"] for row in suite_rows) / len(suite_rows),
            "delta_macro_rate": sum(row["delta_rate"] for row in suite_rows) / len(suite_rows),
            "paired": sum_paired_counts(suite_rows),
        }

    per_seed_rows = []
    for seed in base_results["shared_valid_seeds"]:
        base_successes = 0
        adapter_successes = 0
        improved = 0
        regressed = 0
        for view_id, base_view in base_by_view.items():
            adapter_view = adapter_by_view[view_id]
            base_episode = next(
                episode for episode in base_view["episode_results"] if episode["seed"] == seed
            )
            adapter_episode = next(
                episode
                for episode in adapter_view["episode_results"]
                if episode["seed"] == seed
            )
            base_success = bool(base_episode["success"])
            adapter_success = bool(adapter_episode["success"])
            base_successes += int(base_success)
            adapter_successes += int(adapter_success)
            improved += int(not base_success and adapter_success)
            regressed += int(base_success and not adapter_success)
        per_seed_rows.append(
            {
                "seed": seed,
                "views": len(base_by_view),
                "base_successes": base_successes,
                "adapter_successes": adapter_successes,
                "delta_successes": adapter_successes - base_successes,
                "improved": improved,
                "regressed": regressed,
            }
        )

    with (run_dir / "per_seed_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=tuple(per_seed_rows[0]))
        writer.writeheader()
        writer.writerows(per_seed_rows)

    summary = {
        "base_policy_path": base_results["policy_path"],
        "adapter_policy_path": adapter_results["policy_path"],
        "episodes_per_view": base_results["episodes_per_view"],
        "shared_valid_seeds": base_results["shared_valid_seeds"],
        "policy_inference_seed": base_results["policy_inference_seed"],
        "overall": {
            "views": len(rows),
            "episodes": sum(row["episodes"] for row in rows),
            "base_macro_rate": sum(row["base_rate"] for row in rows) / len(rows),
            "adapter_macro_rate": sum(row["adapter_rate"] for row in rows) / len(rows),
            "delta_macro_rate": sum(row["delta_rate"] for row in rows) / len(rows),
            "paired": sum_paired_counts(rows),
        },
        "by_suite": suite_summary,
        "per_seed": per_seed_rows,
    }
    with (run_dir / "comparison.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        summary_file.write("\n")


def main() -> None:
    cli = parse_args()
    base_path = cli.base_policy_path.expanduser().resolve()
    adapter_path = cli.adapter_policy_path.expanduser().resolve()
    for path in (base_path, adapter_path):
        if not path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    run_name = cli.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cli.output_dir.expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    base_dir = run_dir / "no_adapter"
    adapter_dir = run_dir / "with_adapter"
    print("\n=== Unseen-view baseline: starting ===", flush=True)
    run_evaluator(cli, base_path, base_dir, adapter_enabled=False)
    print("\n=== Unseen-view Adapter: starting ===", flush=True)
    run_evaluator(cli, adapter_path, adapter_dir, adapter_enabled=True)
    write_comparison(
        run_dir,
        load_results(base_dir / "results.json"),
        load_results(adapter_dir / "results.json"),
    )
    print(f"\nSaved unseen-view A/B comparison to {run_dir}")


if __name__ == "__main__":
    main()
