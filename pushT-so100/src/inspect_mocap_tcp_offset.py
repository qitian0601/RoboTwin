from pathlib import Path

import mujoco
import numpy as np
import pandas as pd


XML_PATH = "chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/nero_human_env.xml"
DATA_PATH = Path("data/nero-dataset/data/chunk-000/file-000.parquet")


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    qaddrs = [model.jnt_qposadr[joint_id] for joint_id in joint_ids]

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
    mocap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_mocap")
    mocap_id = model.body(mocap_body_id).mocapid[0]

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    print("RESET")
    print("mocap_pos       ", np.round(data.mocap_pos[mocap_id], 6))
    print("gripper_base    ", np.round(data.xpos[base_id], 6))
    print("gripper_tcp     ", np.round(data.site_xpos[tcp_id], 6))
    print("mocap - tcp     ", np.round(data.mocap_pos[mocap_id] - data.site_xpos[tcp_id], 6))
    print("mocap - base    ", np.round(data.mocap_pos[mocap_id] - data.xpos[base_id], 6))
    print("tcp - base      ", np.round(data.site_xpos[tcp_id] - data.xpos[base_id], 6))

    df = pd.read_parquet(DATA_PATH)
    print("\nSAMPLES")
    for idx in [0, 1, 20, 50, 100, 150, min(200, len(df) - 1)]:
        state = np.asarray(df["observation.state"].iloc[idx], dtype=np.float64)
        action = np.asarray(df["action"].iloc[idx], dtype=np.float64)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        for qaddr, value in zip(qaddrs, state[:7], strict=True):
            data.qpos[qaddr] = value
        mujoco.mj_forward(model, data)

        base = data.xpos[base_id].copy()
        tcp = data.site_xpos[tcp_id].copy()
        print(f"{idx} frame={int(df['frame_index'].iloc[idx])}")
        print("  action xyz        ", np.round(action[:3], 6))
        print("  FK base xyz       ", np.round(base, 6))
        print("  action - base     ", np.round(action[:3] - base, 6), np.linalg.norm(action[:3] - base))
        print("  FK tcp xyz        ", np.round(tcp, 6))
        print("  action - tcp      ", np.round(action[:3] - tcp, 6), np.linalg.norm(action[:3] - tcp))
        print("  tcp - base        ", np.round(tcp - base, 6), np.linalg.norm(tcp - base))

    actions = []
    bases = []
    tcps = []
    count = 0
    for parquet_file in sorted(Path("data/nero-dataset/data").glob("chunk-*/*.parquet")):
        df = pd.read_parquet(parquet_file)
        for _, row in df.iterrows():
            state = np.asarray(row["observation.state"], dtype=np.float64)
            action = np.asarray(row["action"], dtype=np.float64)
            mujoco.mj_resetDataKeyframe(model, data, 0)
            for qaddr, value in zip(qaddrs, state[:7], strict=True):
                data.qpos[qaddr] = value
            mujoco.mj_forward(model, data)
            actions.append(action[:3])
            bases.append(data.xpos[base_id].copy())
            tcps.append(data.site_xpos[tcp_id].copy())
            count += 1
            if count >= 1500:
                break
        if count >= 1500:
            break

    actions = np.asarray(actions)
    bases = np.asarray(bases)
    tcps = np.asarray(tcps)
    for name, values in [
        ("action-base", actions - bases),
        ("action-tcp", actions - tcps),
        ("tcp-base", tcps - bases),
    ]:
        norms = np.linalg.norm(values, axis=1)
        print(f"\nAGG {name}")
        print("  mean diff          ", np.round(values.mean(axis=0), 6))
        print("  std diff           ", np.round(values.std(axis=0), 6))
        print(
            "  norm mean/std/min/max",
            float(norms.mean()),
            float(norms.std()),
            float(norms.min()),
            float(norms.max()),
        )


if __name__ == "__main__":
    main()

