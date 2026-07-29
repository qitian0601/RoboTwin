#!/usr/bin/env python3
"""Select a candidate Adapter without allowing canonical-view regressions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canonical-tolerance", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.aggregate.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No evaluation rows found in {args.aggregate}")

    by_policy: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_policy.setdefault(row["policy_label"], {})[row["view"]] = row
    if "old_base" not in by_policy:
        raise RuntimeError("Aggregate results must contain old_base")

    def rate(policy: str, view: str) -> float:
        try:
            return float(by_policy[policy][view]["success_rate"])
        except KeyError as exc:
            raise RuntimeError(f"Missing {policy}/{view} result") from exc

    baseline_c0 = rate("old_base", "C0")
    labels = [label for label in by_policy if label != "old_base"]
    records = []
    for label in labels:
        canonical = rate(label, "C0")
        shifted_rates = [rate(label, f"C{i}") for i in range(1, 7)]
        record = {
            "label": label,
            "canonical_rate": canonical,
            "canonical_delta": canonical - baseline_c0,
            "canonical_ok": canonical >= baseline_c0 - args.canonical_tolerance,
            "shifted_macro_rate": sum(shifted_rates) / len(shifted_rates),
            "shifted_rates": {f"C{i}": shifted_rates[i - 1] for i in range(1, 7)},
        }
        record["status"] = "candidate" if record["canonical_ok"] else "rejected_canonical_regression"
        records.append(record)

    stable_record = next(
        (record for record in records if record["label"] == "teacher_first_adapter"), None
    )
    eligible = [record for record in records if record["canonical_ok"]]
    # A candidate must show a strict shifted-view improvement to replace the
    # established teacher-first checkpoint.  Prefer the stable checkpoint on
    # an exact tie instead of allowing directory/CSV ordering to decide which
    # policy is deployed.
    selected = (
        max(
            eligible,
            key=lambda record: (
                record["shifted_macro_rate"],
                record["label"] == "teacher_first_adapter",
            ),
        )
        if eligible
        else None
    )
    rollback_label = (
        "teacher_first_adapter"
        if stable_record is not None and stable_record["canonical_ok"]
        else "old_base"
    )
    report = {
        "baseline": {"label": "old_base", "canonical_rate": baseline_c0},
        "canonical_tolerance": args.canonical_tolerance,
        "variants": records,
        "selected": selected,
        "stable_policy": stable_record,
        "rollback_policy": rollback_label,
        "rollback_reason": (
            "teacher_first_adapter passed the canonical 5pp gate"
            if rollback_label == "teacher_first_adapter"
            else "teacher_first_adapter is missing or failed the canonical 5pp gate; use old_base"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
