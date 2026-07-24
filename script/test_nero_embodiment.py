import sapien.core as sapien


def main():
    engine = sapien.Engine()
    scene = engine.create_scene()
    scene.set_timestep(1 / 250)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    urdf_path = "./assets/embodiments/nero/nero_with_gripper_description.urdf"

    left = loader.load(urdf_path)
    right = loader.load(urdf_path)
    left.set_root_pose(sapien.Pose([-0.4, -0.65, 0.74], [1, 0, 0, 0]))
    right.set_root_pose(sapien.Pose([0.4, -0.65, 0.74], [1, 0, 0, 0]))

    homestate = [0.0, 0.0, 0.0, 0.565, 0.0, 0.110, 0.0]
    arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    for entity in (left, right):
        for name, target in zip(arm_joint_names, homestate):
            joint = entity.find_joint_by_name(name)
            joint.set_drive_property(stiffness=1000, damping=200)
            joint.set_drive_target(target)

        entity.find_joint_by_name("gripper_joint1").set_drive_target(0.05)
        entity.find_joint_by_name("gripper_joint2").set_drive_target(-0.05)

    for _ in range(20):
        scene.step()

    print("left active joints:", [joint.get_name() for joint in left.get_active_joints()])
    print("right active joints:", [joint.get_name() for joint in right.get_active_joints()])
    print("left links:", [link.get_name() for link in left.get_links()])
    print("right links:", [link.get_name() for link in right.get_links()])
    print("Nero dual-arm embodiment load ok")


if __name__ == "__main__":
    main()
