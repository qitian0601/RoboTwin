#!/usr/bin/env python3
"""Validate and summarize a paired base/Adapter camera-view evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASKS = ("place_two_cubes_box", "beat_block_hammer")
VIEWS = tuple(f"C{index}" for index in range(7))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--adapter-run", type=Path, required=True)
    parser.add_argument("--episodes-per-view", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def load_run(run_dir: Path, episodes: int) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for task in TASKS:
        path = run_dir / task / "results.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        seeds = payload.get("shared_valid_seeds")
        if not isinstance(seeds, list) or len(seeds) != episodes:
            raise ValueError(f"{path}: expected {episodes} shared seeds")
        by_view = {view["id"]: view for view in payload.get("views", [])}
        if set(by_view) != set(VIEWS):
            raise ValueError(f"{path}: expected complete C0-C6 views")
        for view_id in VIEWS:
            view = by_view[view_id]
            episode_results = view.get("episode_results", [])
            if not view.get("complete") or len(episode_results) != episodes:
                raise ValueError(f"{path}: {view_id} is incomplete")
            if [item.get("seed") for item in episode_results] != seeds:
                raise ValueError(f"{path}: {view_id} does not use shared seeds")
            for item in episode_results:
                video = item.get("video")
                if item.get("success") and video is not None:
                    raise ValueError(f"{path}: successful episode retained a video")
                if not item.get("success"):
                    if not video or not Path(video).is_file():
                        raise ValueError(f"{path}: failed episode has no video")
        results[task] = payload
    return results


def episode_map(payload: dict, view_id: str) -> dict[int, bool]:
    view = next(item for item in payload["views"] if item["id"] == view_id)
    return {int(item["seed"]): bool(item["success"]) for item in view["episode_results"]}


def main() -> None:
    cli = parse_args()
    base_dir = cli.base_run.expanduser().resolve()
    adapter_dir = cli.adapter_run.expanduser().resolve()
    base = load_run(base_dir, cli.episodes_per_view)
    adapter = load_run(adapter_dir, cli.episodes_per_view)
    for task in TASKS:
        if base[task]["shared_valid_seeds"] != adapter[task]["shared_valid_seeds"]:
            raise ValueError(f"{task}: base and Adapter scenario seeds differ")
    if cli.validate_only:
        return

    rows = []
    for task in TASKS:
        for view_id in VIEWS:
            base_outcomes = episode_map(base[task], view_id)
            adapter_outcomes = episode_map(adapter[task], view_id)
            base_successes = sum(base_outcomes.values())
            adapter_successes = sum(adapter_outcomes.values())
            improved = sum(
                not base_outcomes[seed] and adapter_outcomes[seed]
                for seed in base_outcomes
            )
            regressed = sum(
                base_outcomes[seed] and not adapter_outcomes[seed]
                for seed in base_outcomes
            )
            rows.append(
                {
                    "task": task,
                    "view": view_id,
                    "episodes": cli.episodes_per_view,
                    "base_successes": base_successes,
                    "base_success_rate": base_successes / cli.episodes_per_view,
                    "adapter_successes": adapter_successes,
                    "adapter_success_rate": adapter_successes / cli.episodes_per_view,
                    "delta_percentage_points": 100.0
                    * (adapter_successes - base_successes)
                    / cli.episodes_per_view,
                    "paired_improved": improved,
                    "paired_regressed": regressed,
                }
            )

    output_dir = cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    shifted = [row for row in rows if row["view"] != "C0"]
    base_shifted = sum(row["base_successes"] for row in shifted)
    adapter_shifted = sum(row["adapter_successes"] for row in shifted)
    shifted_total = sum(row["episodes"] for row in shifted)
    report = [
        "# PI0.5 24k vs Teacher-First Adapter 4.5k",
        "",
        f"- Base run: `{base_dir}`",
        f"- Adapter run: `{adapter_dir}`",
        f"- Episodes: {cli.episodes_per_view} per task/view, paired scenario seeds",
        "- Videos: failures only",
        "- Pickplace: default task success criterion and standard closed-loop control",
        (
            "- Hammer: observed slow120 training support, physical hammer/block "
            "contact success, execute 40 of 50 predicted actions, new action weight 1.0"
        ),
        "",
        "| Task | View | Base | Adapter | Delta (pp) | Improved | Regressed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['task']} | {row['view']} | "
            f"{row['base_successes']}/{row['episodes']} | "
            f"{row['adapter_successes']}/{row['episodes']} | "
            f"{row['delta_percentage_points']:+.1f} | "
            f"{row['paired_improved']} | {row['paired_regressed']} |"
        )
    report.extend(
        [
            "",
            "## Shifted-view aggregate",
            "",
            f"- Base C1-C6: {base_shifted}/{shifted_total} ({base_shifted / shifted_total:.1%})",
            (
                f"- Adapter C1-C6: {adapter_shifted}/{shifted_total} "
                f"({adapter_shifted / shifted_total:.1%})"
            ),
            (
                "- Difference: "
                f"{100.0 * (adapter_shifted - base_shifted) / shifted_total:+.1f} percentage points"
            ),
            "",
            "`paired_improved` and `paired_regressed` compare the two policies on the exact same scene seed.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
