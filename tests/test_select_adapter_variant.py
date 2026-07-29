from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "script/select_adapter_variant.py"


def test_selector_rejects_canonical_regression_and_picks_best_eligible(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate.csv"
    rows = []
    labels = {
        "old_base": (1.0, [0.5] * 6),
        "teacher_first_adapter": (0.96, [0.6] * 6),
        "pure_teacher_gated_residual": (0.94, [0.9] * 6),
        "multi_scale_dynamic": (1.0, [0.7] * 6),
        "image_routed_moe": (0.98, [0.65] * 6),
    }
    for label, (c0, shifted) in labels.items():
        rows.append({"policy_label": label, "view": "C0", "success_rate": str(c0)})
        for index, value in enumerate(shifted, start=1):
            rows.append({"policy_label": label, "view": f"C{index}", "success_rate": str(value)})
    with aggregate.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("policy_label", "view", "success_rate"))
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "selection.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--aggregate", str(aggregate), "--output", str(output)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    statuses = {record["label"]: record["status"] for record in report["variants"]}
    assert statuses["pure_teacher_gated_residual"] == "rejected_canonical_regression"
    assert report["selected"]["label"] == "multi_scale_dynamic"
    assert report["rollback_policy"] == "teacher_first_adapter"
    assert "multi_scale_dynamic" in completed.stdout


def test_selector_keeps_stable_checkpoint_on_shifted_macro_tie(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate.csv"
    rows = []
    # Put the new candidate before teacher_first_adapter to prove selection is
    # independent of CSV/directory ordering.
    labels = {
        "old_base": (0.95, [0.50] * 6),
        "multi_scale_dynamic": (0.95, [0.70] * 6),
        "teacher_first_adapter": (0.95, [0.70] * 6),
    }
    for label, (c0, shifted) in labels.items():
        rows.append({"policy_label": label, "view": "C0", "success_rate": str(c0)})
        for index, value in enumerate(shifted, start=1):
            rows.append({"policy_label": label, "view": f"C{index}", "success_rate": str(value)})
    with aggregate.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("policy_label", "view", "success_rate"))
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "selection.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--aggregate", str(aggregate), "--output", str(output)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["selected"]["label"] == "teacher_first_adapter"
