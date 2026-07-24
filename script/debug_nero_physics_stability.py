import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
import sapien.core as sapien
import yaml


def load_config(repo_root):
    cfg_path = os.path.join(repo_root, "assets", "embodiments", "nero", "config.yml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_collision_variant_urdf(urdf_path, mode):
    if mode == "original":
        return urdf_path

    remove_links = {
        "no_body": {
            "base_link",
            "link1",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "link7",
        },
        "no_arm": {
            "base_link",
            "link1",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "link7",
            "gripper_flange",
            "gripper_base",
        },
        "no_all": {
            "base_link",
            "link1",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "link7",
            "gripper_flange",
            "gripper_base",
            "gripper_link1",
            "gripper_link2",
        },
    }[mode]

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall("link"):
        if link.get("name") not in remove_links:
            continue
        for collision in list(link.findall("collision")):
            link.remove(collision)

    out_path = os.path.join(os.path.dirname(urdf_path), f"nero_{mode}_collision.urdf")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def make_scene():
    engine = sapien.Engine()
    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 250)
    scene.create_physical_material(1, 0.5, 0.01)
    return scene


def load_arm(scene, urdf_path, srdf_path, pose, use_srdf):
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    if use_srdf and srdf_path:
        with open(srdf_path, "r", encoding="utf-8") as f:
            loader.parse_srdf(f.read())
    arm = loader.load(urdf_path)
    arm.set_root_pose(sapien.Pose(pose[:3], pose[-4:]))
    return arm


def set_drive_and_qpos(arm, arm_qpos):
    names = [j.get_name() for j in arm.get_active_joints()]
    qpos = np.zeros(len(names), dtype=np.float32)
    qpos[: len(arm_qpos)] = arm_qpos
    arm.set_qpos(qpos)
    arm.set_qvel(np.zeros_like(qpos))
    for joint, target in zip(arm.get_active_joints(), qpos):
        joint.set_drive_property(1000, 200)
        joint.set_drive_target(float(target))
    return names


def contact_pairs(scene):
    pairs = set()
    for contact in scene.get_contacts():
        a0 = contact.bodies[0].entity.name
        a1 = contact.bodies[1].entity.name
        pairs.add(tuple(sorted((a0, a1))))
    return sorted(pairs)


def check_case(repo_root, collision_mode, use_srdf, force_mass):
    cfg = load_config(repo_root)
    nero_dir = os.path.join(repo_root, "assets", "embodiments", "nero")
    urdf_path = make_collision_variant_urdf(
        os.path.join(nero_dir, "nero_with_gripper_description.urdf"),
        collision_mode,
    )
    srdf_name = cfg.get("srdf_path")
    srdf_path = os.path.join(nero_dir, srdf_name) if srdf_name else None
    pose0 = cfg["robot_pose"][0].copy()
    pose1 = cfg["robot_pose"][1].copy()
    arms_dis = 0.8
    pose0[:3] = [pose0[0] - arms_dis / 2, pose0[1], pose0[2]]
    pose1[:3] = [pose1[0] + arms_dis / 2, pose1[1], pose1[2]]

    scene = make_scene()
    left = load_arm(scene, urdf_path, srdf_path, pose0, use_srdf)
    right = load_arm(scene, urdf_path, srdf_path, pose1, use_srdf)

    if force_mass:
        for arm in (left, right):
            for link in arm.get_links():
                link.set_mass(1)

    names = set_drive_and_qpos(left, cfg["homestate"][0])
    set_drive_and_qpos(right, cfg["homestate"][1])
    lower = np.array([j.get_limits()[0, 0] for j in left.get_active_joints()])
    upper = np.array([j.get_limits()[0, 1] for j in left.get_active_joints()])

    first_bad = None
    first_contacts = None
    for step in range(300):
        scene.step()
        lq = left.get_qpos()
        rq = right.get_qpos()
        bad = (
            np.where((lq < lower - 1e-3) | (lq > upper + 1e-3))[0].tolist(),
            np.where((rq < lower - 1e-3) | (rq > upper + 1e-3))[0].tolist(),
        )
        pairs = contact_pairs(scene)
        if pairs and first_contacts is None:
            first_contacts = (step, pairs)
        if (bad[0] or bad[1]) and first_bad is None:
            first_bad = (step, bad, lq.copy(), rq.copy())
            break

    print(f"\ncase collision_mode={collision_mode} use_srdf={use_srdf} force_mass={force_mass}")
    print("active joints:", names)
    if first_contacts:
        print("first contacts at step:", first_contacts[0])
        for pair in first_contacts[1]:
            print("  contact:", pair[0], "<->", pair[1])
    else:
        print("first contacts: none")
    if first_bad:
        step, bad, lq, rq = first_bad
        print("first out-of-limit at step:", step)
        print("bad indices left/right:", bad)
        print("left qpos:", np.round(lq, 6).tolist())
        print("right qpos:", np.round(rq, 6).tolist())
    else:
        print("no out-of-limit within 300 steps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=os.getcwd())
    args = parser.parse_args()
    for collision_mode in ("original", "no_body", "no_arm", "no_all"):
        for use_srdf in (False, True):
            for force_mass in (False, True):
                check_case(args.repo_root, collision_mode, use_srdf, force_mass)


if __name__ == "__main__":
    main()
