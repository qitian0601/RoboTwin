"""Pure, CPU-only persistence contract for resumable camera evaluations."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


_RESUME_CORE_FIELDS = (
    "task_name",
    "task_config",
    "policy_path",
    "server_address",
    "instruction",
    "episodes_per_view",
    "max_steps",
    "shared_valid_seeds",
    "policy_inference_seed",
    "adapter_c0_bypassed",
)

_RESUME_VERSIONED_FIELDS = (
    "seed_search_index",
    "scenario_episode_index",
    "scenario_sampling_range",
    "success_criterion",
    "control_config",
    "recording_config",
    "instruction_by_arm",
    "trace_dir",
)


def retain_episode_video(
    video_path: Path | None,
    *,
    success: bool,
    failures_only: bool,
) -> Path | None:
    """Keep failures, deleting only the current successful episode video."""
    if video_path is None:
        return None
    if failures_only and success:
        video_path.unlink(missing_ok=True)
        return None
    return video_path


def write_results(output_dir: Path, payload: dict[str, Any]) -> None:
    """Atomically checkpoint JSON and its human-readable CSV projection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_dir,
        prefix=".results.",
        suffix=".json.tmp",
        delete=False,
    ) as result_file:
        json_tmp = Path(result_file.name)
        json.dump(payload, result_file, ensure_ascii=False, indent=2)
        result_file.write("\n")
        result_file.flush()
        os.fsync(result_file.fileno())
    os.replace(json_tmp, json_path)

    csv_path = output_dir / "summary.csv"
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        dir=output_dir,
        prefix=".summary.",
        suffix=".csv.tmp",
        delete=False,
    ) as csv_file:
        csv_tmp = Path(csv_file.name)
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("view", "description", "successes", "episodes", "success_rate"),
        )
        writer.writeheader()
        for result in payload["views"]:
            writer.writerow(
                {
                    "view": result["id"],
                    "description": result["description"],
                    "successes": result["successes"],
                    "episodes": result["episodes"],
                    "success_rate": result["success_rate"],
                }
            )
        csv_file.flush()
        os.fsync(csv_file.fileno())
    os.replace(csv_tmp, csv_path)


def _view_specs_by_id(view_specs: Iterable[Any]) -> dict[str, Any]:
    mapping = {spec.identifier: spec for spec in view_specs}
    if len(mapping) != 7:
        raise ValueError("Resume contract requires exactly the C0-C6 view specifications")
    return mapping


def _normalise_resumed_views(
    payload: dict[str, Any],
    valid_seeds: list[int],
    view_specs: Iterable[Any],
) -> None:
    """Validate episode checkpoints and upgrade pre-resume result files in memory."""
    specs_by_id = _view_specs_by_id(view_specs)
    views = payload.get("views")
    if not isinstance(views, list):
        raise ValueError("Existing results.json has no valid views list")
    seen: set[str] = set()
    for view in views:
        if not isinstance(view, dict):
            raise ValueError("Existing views entries must be JSON objects")
        identifier = view.get("id")
        if not isinstance(identifier, str):
            raise ValueError("Existing view entry has no string id")
        if identifier in seen:
            raise ValueError(f"Duplicate view id in existing result: {identifier}")
        seen.add(identifier)
        if identifier not in specs_by_id:
            raise ValueError(f"Unknown view id in existing result: {identifier!r}")
        spec = specs_by_id[identifier]
        for field, expected_value in (
            ("kind", spec.kind),
            ("value", spec.value),
            ("description", spec.description),
        ):
            if view.get(field) != expected_value:
                raise ValueError(
                    f"Resume contract mismatch for {identifier}.{field}: "
                    f"existing={view.get(field)!r}, requested={expected_value!r}"
                )
        episodes = view.get("episode_results")
        if not isinstance(episodes, list):
            raise ValueError(f"Existing {identifier} has no episode_results list")
        if len(episodes) > len(valid_seeds):
            raise ValueError(f"Existing {identifier} contains too many episodes")
        episode_seeds = [item.get("seed") for item in episodes]
        expected_prefix = valid_seeds[: len(episodes)]
        if episode_seeds != expected_prefix:
            raise ValueError(
                f"Existing {identifier} episode seeds are not the requested seed prefix: "
                f"{episode_seeds!r} != {expected_prefix!r}"
            )
        if any(not isinstance(item.get("success"), bool) for item in episodes):
            raise ValueError(f"Existing {identifier} has a non-boolean success value")
        successes = sum(int(item["success"]) for item in episodes)
        if int(view.get("successes", -1)) != successes:
            raise ValueError(f"Existing {identifier} successes counter is inconsistent")
        if int(view.get("episodes", -1)) != len(episodes):
            raise ValueError(f"Existing {identifier} episodes counter is inconsistent")
        expected_rate = successes / len(episodes) if episodes else None
        existing_rate = view.get("success_rate")
        if expected_rate is None:
            if existing_rate is not None:
                raise ValueError(f"Existing {identifier} empty success_rate must be null")
        elif existing_rate is None or not math.isclose(
            float(existing_rate), expected_rate, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"Existing {identifier} success_rate is inconsistent")
        inferred_complete = len(episodes) == len(valid_seeds)
        if view.get("complete", inferred_complete) not in (False, True):
            raise ValueError(f"Existing {identifier} complete flag is not boolean")
        if view.get("complete") is True and not inferred_complete:
            raise ValueError(f"Existing {identifier} is marked complete but is partial")
        view["complete"] = inferred_complete


def load_or_create_results(
    output_dir: Path,
    expected: dict[str, Any],
    *,
    resume_existing: bool,
    view_specs: Iterable[Any],
) -> dict[str, Any]:
    """Load a resumable result after strict validation, or create a fresh payload."""
    results_path = output_dir / "results.json"
    if not results_path.exists():
        if output_dir.exists() and any(output_dir.iterdir()) and not resume_existing:
            raise FileExistsError(
                f"Refusing to write a fresh evaluation into non-empty directory: {output_dir}"
            )
        return {**expected, "resume_contract_version": 1, "views": []}
    if not resume_existing:
        raise FileExistsError(
            f"Evaluation results already exist; use --resume-existing to validate and resume: "
            f"{results_path}"
        )
    with results_path.open(encoding="utf-8") as result_file:
        payload = json.load(result_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Existing evaluation result is not a JSON object: {results_path}")
    for field in _RESUME_CORE_FIELDS:
        if payload.get(field) != expected[field]:
            raise ValueError(
                f"Resume contract mismatch for {field}: "
                f"existing={payload.get(field)!r}, requested={expected[field]!r}"
            )
    # Older files predate these explicit fields. Upgrade them only after every
    # historically recorded, result-affecting field matched above.
    for field in _RESUME_VERSIONED_FIELDS:
        requested = expected.get(field)
        if field in payload and payload[field] != requested:
            raise ValueError(
                f"Resume contract mismatch for {field}: "
                f"existing={payload[field]!r}, requested={requested!r}"
            )
        payload[field] = requested
    payload["resume_contract_version"] = 1
    _normalise_resumed_views(payload, expected["shared_valid_seeds"], view_specs)
    return payload


def existing_view(payload: dict[str, Any], identifier: str) -> dict[str, Any] | None:
    return next((view for view in payload["views"] if view["id"] == identifier), None)


def unused_video_path(base_path: Path) -> Path:
    """Never overwrite a video left by an interrupted, uncheckpointed episode."""
    if not base_path.exists():
        return base_path
    retry = 1
    while True:
        candidate = base_path.with_name(f"{base_path.stem}_retry{retry}{base_path.suffix}")
        if not candidate.exists():
            return candidate
        retry += 1
