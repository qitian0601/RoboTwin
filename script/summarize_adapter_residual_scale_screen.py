#!/usr/bin/env python3
"""Summarize the conservative Adapter residual-scale closed-loop screen."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CASES = (
    ("place_two_cubes_box", "C1"),
    ("place_two_cubes_box", "C4"),
    ("beat_block_hammer", "C3"),
    ("beat_block_hammer", "C5"),
)


def outcomes(path: Path, expected_seeds: list[int]) -> list[bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    views = payload.get("views", [])
    if len(views) != 1 or not views[0].get("complete"):
        raise ValueError(f"Incomplete screen result: {path}")
    episodes = views[0].get("episode_results", [])
    if [item.get("seed") for item in episodes] != expected_seeds:
        raise ValueError(f"Seed mismatch: {path}")
    return [bool(item["success"]) for item in episodes]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    args = parser.parse_args()
    root = args.screen_root.expanduser().resolve()
    reference = args.reference_run.expanduser().resolve()
    labels = ("alpha_0p10", "alpha_0p25", "alpha_0p50")

    reference_payloads = {}
    for policy in ("base_step024000", "adapter_step004500"):
        reference_payloads[policy] = {
            task: json.loads((reference / policy / task / "results.json").read_text())
            for task, _ in CASES
        }

    rows = []
    totals = {"base": 0, "alpha_1p00": 0, **{label: 0 for label in labels}}
    for task, view_id in CASES:
        base_payload = reference_payloads["base_step024000"][task]
        adapter_payload = reference_payloads["adapter_step004500"][task]
        start_index = 14 if task == "place_two_cubes_box" else 0
        seeds = base_payload["shared_valid_seeds"][start_index : start_index + 5]
        base_view = next(view for view in base_payload["views"] if view["id"] == view_id)
        adapter_view = next(view for view in adapter_payload["views"] if view["id"] == view_id)
        base = [item["success"] for item in base_view["episode_results"]][
            start_index : start_index + 5
        ]
        alpha_one = [item["success"] for item in adapter_view["episode_results"]][
            start_index : start_index + 5
        ]
        row = {
            "task": task,
            "view": view_id,
            "episodes": 5,
            "base": sum(base),
            "alpha_1p00": sum(alpha_one),
        }
        totals["base"] += row["base"]
        totals["alpha_1p00"] += row["alpha_1p00"]
        for label in labels:
            result_path = root / label / task / view_id / "results.json"
            result = outcomes(result_path, seeds)
            row[label] = sum(result)
            totals[label] += row[label]
        rows.append(row)

    with (root / "screen_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    candidate_labels = list(labels)
    winner = max(candidate_labels, key=lambda label: (totals[label], -float(label[6:].replace("p", "."))))
    deploy = totals[winner] >= totals["base"]
    report = [
        "# Adapter Residual-Scale Closed-Loop Screen",
        "",
        "The 20 cases are fixed scenes where the unmodified Adapter showed clear regressions.",
        "No screening episode was used to train any candidate.",
        "",
        "| Candidate | Successes | Rate |",
        "|---|---:|---:|",
        f"| Base 24k (no Adapter) | {totals['base']}/20 | {totals['base']/20:.1%} |",
        f"| Adapter alpha=1.00 | {totals['alpha_1p00']}/20 | {totals['alpha_1p00']/20:.1%} |",
    ]
    for label in labels:
        alpha = label.removeprefix("alpha_").replace("p", ".")
        report.append(f"| Adapter alpha={alpha} | {totals[label]}/20 | {totals[label]/20:.1%} |")
    report.extend(
        [
            "",
            f"Best scaled candidate: `{winner}`.",
            (
                "Deployment decision: eligible for broader validation."
                if deploy
                else "Deployment decision: reject; it does not match the no-Adapter base."
            ),
            "",
            "The original alpha=1 checkpoint and base model were not modified.",
            "",
        ]
    )
    (root / "SCREEN_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (root / "screen_selection.json").write_text(
        json.dumps(
            {"totals": totals, "winner": winner, "eligible_for_broader_validation": deploy},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
