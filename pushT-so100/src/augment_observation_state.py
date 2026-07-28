import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import mujoco

from state_features import JOINT_STATE_NAMES, OBS_STATE_NAMES
from tcp_frame import (
    compute_mocap_to_tcp_transform,
    get_site_pose,
    mocap_pose_to_tcp_pose,
    pack_pose,
)


QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert observation.state between joint-only and joint+EE formats."
    )
    parser.add_argument("--dataset-path", type=Path, default=Path("data/nero-dataset"))
    parser.add_argument(
        "--mode",
        choices=["prev_action", "same_action", "mujoco_tcp", "revert_7d"],
        default="prev_action",
        help=(
            "prev_action appends the previous frame action as current EE state; "
            "same_action appends the same-frame action; "
            "mujoco_tcp converts old mocap action to TCP action and state; "
            "revert_7d restores joint-only state."
        ),
    )
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=Path("chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/nero_human_env.xml"),
        help="MuJoCo XML used for mujoco_tcp conversion.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def to_float32_array(value):
    return np.asarray(value, dtype=np.float32)


def joint_state_from(state):
    state = to_float32_array(state)
    if state.shape != (len(JOINT_STATE_NAMES),):
        if state.shape == (len(OBS_STATE_NAMES),):
            return state[: len(JOINT_STATE_NAMES)].astype(np.float32)
        raise ValueError(f"Unsupported observation.state shape: {state.shape}")
    return state.astype(np.float32)


def augment_state(state, action):
    state = joint_state_from(state)
    action = to_float32_array(action)
    if action.shape != (8,):
        raise ValueError(f"Unsupported action shape: {action.shape}")
    return np.concatenate([state, action]).astype(np.float32)


def stats_from_array(values):
    stats = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [float(values.shape[0])],
    }
    for key, q in QUANTILES.items():
        stats[key] = np.quantile(values, q, axis=0).tolist()
    return stats


def update_info_json(dataset_path):
    info_path = dataset_path / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    info["features"]["observation.state"] = {
        "dtype": "float32",
        "shape": [len(OBS_STATE_NAMES)],
        "names": OBS_STATE_NAMES,
    }
    info["features"]["action"] = {
        "dtype": "float32",
        "shape": [8],
        "names": [
            "tcp_x",
            "tcp_y",
            "tcp_z",
            "tcp_qw",
            "tcp_qx",
            "tcp_qy",
            "tcp_qz",
            "gripper",
        ],
    }

    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
        f.write("\n")


def update_info_json_revert(dataset_path):
    info_path = dataset_path / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    info["features"]["observation.state"] = {
        "dtype": "float32",
        "shape": [len(JOINT_STATE_NAMES)],
        "names": JOINT_STATE_NAMES,
    }
    info["features"]["action"] = {
        "dtype": "float32",
        "shape": [8],
        "names": [
            "mocap_x",
            "mocap_y",
            "mocap_z",
            "mocap_qw",
            "mocap_qx",
            "mocap_qy",
            "mocap_qz",
            "gripper",
        ],
    }

    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
        f.write("\n")


def update_stats_json(dataset_path, state_values):
    stats_path = dataset_path / "meta" / "stats.json"
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    stats["observation.state"] = stats_from_array(state_values)

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
        f.write("\n")


def update_stats_json_with_action(dataset_path, state_values, action_values):
    stats_path = dataset_path / "meta" / "stats.json"
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    stats["observation.state"] = stats_from_array(state_values)
    stats["action"] = stats_from_array(action_values)

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
        f.write("\n")


def make_mujoco_context(xml_path):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in JOINT_STATE_NAMES
    ]
    qaddrs = [model.jnt_qposadr[joint_id] for joint_id in joint_ids]
    tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
    mocap_id = model.body("target_mocap").mocapid[0]
    mocap_to_tcp = compute_mocap_to_tcp_transform(model, data, mocap_id, tcp_site_id)
    return model, data, qaddrs, tcp_site_id, mocap_to_tcp


def convert_frame_states(df, mode, mujoco_context=None):
    states = [joint_state_from(state) for state in df["observation.state"]]

    if mode == "revert_7d":
        return states, [to_float32_array(action) for action in df["action"]]

    actions = [to_float32_array(action) for action in df["action"]]
    if mode == "mujoco_tcp":
        model, data, qaddrs, tcp_site_id, mocap_to_tcp = mujoco_context
        new_states = []
        new_actions = []
        for state, action in zip(states, actions, strict=True):
            mujoco.mj_resetDataKeyframe(model, data, 0)
            for qaddr, value in zip(qaddrs, state, strict=True):
                data.qpos[qaddr] = value
            mujoco.mj_forward(model, data)
            tcp_pos, tcp_quat = get_site_pose(data, tcp_site_id)
            gripper = float(action[7])
            tcp_state = pack_pose(tcp_pos, tcp_quat, gripper)
            tcp_target_pos, tcp_target_quat = mocap_pose_to_tcp_pose(
                action[:3], action[3:7], mocap_to_tcp
            )
            tcp_action = pack_pose(tcp_target_pos, tcp_target_quat, gripper)
            new_states.append(np.concatenate([state, tcp_state]).astype(np.float32))
            new_actions.append(tcp_action)
        return new_states, new_actions

    new_states = []
    new_actions = actions
    previous_by_episode = {}

    for state, action, episode_index in zip(
        states, actions, df["episode_index"], strict=True
    ):
        episode_key = int(episode_index)
        if mode == "same_action":
            ee_state = action
        else:
            ee_state = previous_by_episode.get(episode_key, action)

        new_states.append(np.concatenate([state, ee_state]).astype(np.float32))
        previous_by_episode[episode_key] = action

    return new_states, new_actions


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    parquet_files = sorted((dataset_path / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_path / 'data'}")

    all_states = []
    all_actions = []
    changed_files = 0
    mujoco_context = make_mujoco_context(args.xml_path) if args.mode == "mujoco_tcp" else None

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        new_states, new_actions = convert_frame_states(df, args.mode, mujoco_context)
        all_states.extend(new_states)
        all_actions.extend(new_actions)

        first_shape = to_float32_array(df["observation.state"].iloc[0]).shape
        target_shape = (len(JOINT_STATE_NAMES),) if args.mode == "revert_7d" else (len(OBS_STATE_NAMES),)
        sample_new = np.stack(new_states[: min(len(new_states), 16)], axis=0)
        sample_old = np.stack(
            [to_float32_array(x) for x in df["observation.state"].iloc[: len(sample_new)]],
            axis=0,
        )
        sample_action_new = np.stack(new_actions[: len(sample_new)], axis=0)
        sample_action_old = np.stack(
            [to_float32_array(x) for x in df["action"].iloc[: len(sample_new)]],
            axis=0,
        )
        if (
            first_shape == target_shape
            and np.allclose(sample_old, sample_new)
            and np.allclose(sample_action_old, sample_action_new)
        ):
            continue

        changed_files += 1
        if not args.dry_run:
            df["observation.state"] = [state.tolist() for state in new_states]
            df["action"] = [action.tolist() for action in new_actions]
            df.to_parquet(parquet_file, index=False)

    state_values = np.stack(all_states, axis=0).astype(np.float32)
    action_values = np.stack(all_actions, axis=0).astype(np.float32)
    if not args.dry_run:
        if args.mode == "revert_7d":
            update_info_json_revert(dataset_path)
            update_stats_json(dataset_path, state_values)
        elif args.mode == "mujoco_tcp":
            update_info_json(dataset_path)
            update_stats_json_with_action(dataset_path, state_values, action_values)
        else:
            update_info_json(dataset_path)
            update_stats_json(dataset_path, state_values)

    mode = "DRY RUN" if args.dry_run else "UPDATED"
    print(f"{mode}: mode={args.mode}, files={len(parquet_files)}, changed_files={changed_files}")
    print(f"observation.state shape: {state_values.shape[1]}")
    print(f"frames: {state_values.shape[0]}")


if __name__ == "__main__":
    main()
