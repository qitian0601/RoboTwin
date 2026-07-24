from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import *
import numpy as np
import sapien


class place_two_cubes_box(Base_Task):

    def setup_demo(self, **kwags):
        self.grasp_center_z_offset_m = float(
            kwags.get("grasp_center_z_offset_m", 0.0)
        )
        success_config = kwags.get("success", {})
        self.success_hold_s = float(success_config.get("hold_s", 0.5))
        self.success_max_linear_velocity = float(
            success_config.get("max_linear_velocity_m_s", 0.05)
        )
        self.success_max_angular_velocity = float(
            success_config.get("max_angular_velocity_rad_s", 1.0)
        )
        self.success_height_tolerance = float(
            success_config.get("height_tolerance_m", 0.005)
        )
        self._success_stable_steps = 0
        super()._init_task_env_(**kwags)

    def _rand_xy(self, base_xy, xy_range=(0.03, 0.03)):
        return np.array([
            base_xy[0] + np.random.uniform(-xy_range[0], xy_range[0]),
            base_xy[1] + np.random.uniform(-xy_range[1], xy_range[1]),
        ])

    def load_actors(self):
        self.table_top_z = 0.74
        self.cube_half = 0.025  # 5 cm cube edge length.

        self.box_length = 0.30
        self.box_width = 0.20
        self.box_height = 0.09
        self.box_half_xy = np.array([self.box_length / 2, self.box_width / 2])
        self.box_center = np.array([0.0, -0.15, self.table_top_z])

        right_xy = self._rand_xy([0.3, -0.3])
        left_xy = self._rand_xy([-0.3, -0.3])

        self.right_cube = create_box(
            scene=self,
            pose=sapien.Pose([right_xy[0], right_xy[1], self.table_top_z + self.cube_half], [1, 0, 0, 0]),
            half_size=(self.cube_half, self.cube_half, self.cube_half),
            color=(1.0, 0.82, 0.05),
            name="yellow_cube",
        )
        self.left_cube = create_box(
            scene=self,
            pose=sapien.Pose([left_xy[0], left_xy[1], self.table_top_z + self.cube_half], [1, 0, 0, 0]),
            half_size=(self.cube_half, self.cube_half, self.cube_half),
            color=(0.0, 0.65, 0.32),
            name="green_cube",
        )
        self._offset_cube_grasp_centers(self.right_cube)
        self._offset_cube_grasp_centers(self.left_cube)
        self.right_cube.set_mass(0.02)
        self.left_cube.set_mass(0.02)

        floor_half_z = 0.005
        wall_half_t = 0.005
        wall_half_h = self.box_height / 2
        wall_center_z = self.table_top_z + floor_half_z * 2 + wall_half_h
        self.box_floor_top_z = self.table_top_z + floor_half_z * 2
        self.box_wall_top_z = wall_center_z + wall_half_h
        cx, cy, _ = self.box_center
        hx, hy = self.box_half_xy

        self.box_floor = create_box(
            scene=self,
            pose=sapien.Pose([cx, cy, self.table_top_z + floor_half_z], [1, 0, 0, 0]),
            half_size=(hx, hy, floor_half_z),
            color=(0.005, 0.005, 0.005),
            name="black_box_floor",
            is_static=True,
        )
        self.box_walls = [
            create_box(self, sapien.Pose([cx - hx - wall_half_t, cy, wall_center_z], [1, 0, 0, 0]),
                       (wall_half_t, hy + wall_half_t, wall_half_h),
                       color=(0.005, 0.005, 0.005), name="black_box_left_wall", is_static=True),
            create_box(self, sapien.Pose([cx + hx + wall_half_t, cy, wall_center_z], [1, 0, 0, 0]),
                       (wall_half_t, hy + wall_half_t, wall_half_h),
                       color=(0.005, 0.005, 0.005), name="black_box_right_wall", is_static=True),
            create_box(self, sapien.Pose([cx, cy - hy - wall_half_t, wall_center_z], [1, 0, 0, 0]),
                       (hx + wall_half_t, wall_half_t, wall_half_h),
                       color=(0.005, 0.005, 0.005), name="black_box_front_wall", is_static=True),
            create_box(self, sapien.Pose([cx, cy + hy + wall_half_t, wall_center_z], [1, 0, 0, 0]),
                       (hx + wall_half_t, wall_half_t, wall_half_h),
                       color=(0.005, 0.005, 0.005), name="black_box_back_wall", is_static=True),
        ]

        self.right_target_xy = np.array([0.10, self.box_center[1]])
        self.left_target_xy = np.array([-0.10, self.box_center[1]])
        self.right_target_pose = sapien.Pose([self.right_target_xy[0], self.right_target_xy[1],
                                              self.table_top_z + self.cube_half + 0.025],
                                             [1, 0, 0, 0])
        self.left_target_pose = sapien.Pose([self.left_target_xy[0], self.left_target_xy[1],
                                             self.table_top_z + self.cube_half + 0.025],
                                            [1, 0, 0, 0])

        self.add_prohibit_area(self.right_cube, padding=0.06)
        self.add_prohibit_area(self.left_cube, padding=0.06)
        self.add_prohibit_area(self.box_floor, padding=0.08)

    def _offset_cube_grasp_centers(self, cube):
        """Move this task's grasp contacts vertically without changing cube geometry."""
        if abs(self.grasp_center_z_offset_m) < 1e-12:
            return
        normalized_offset = self.grasp_center_z_offset_m / self.cube_half
        for contact_pose in cube.config["contact_points_pose"]:
            contact_pose[2][3] += normalized_offset

    def play_once(self):
        right = ArmTag("right")
        left = ArmTag("left")

        self._place_cube(self.right_cube, right, self.right_target_xy)
        self._return_arm_to_home(right)
        self._place_cube(self.left_cube, left, self.left_target_xy)
        self._return_arm_to_home(left)
        self._hold_final_pose(steps=120)

        self.info["info"] = {
            "{A}": "yellow cube",
            "{B}": "green cube",
            "{C}": "black box",
            "{a}": "right",
            "{b}": "left",
        }
        return self.info

    def _place_cube(self, cube, arm_tag, target_xy, lift_z=0.12, lower_z=0.08, max_step=0.06):
        self.move(self.grasp_actor(cube, arm_tag=arm_tag, pre_grasp_dis=0.08, grasp_dis=-0.015))
        self._move_by_chunks(arm_tag, z=lift_z, max_step=max_step)

        cube_xy = cube.get_pose().p[:2]
        move_xy = target_xy - cube_xy
        self._move_by_chunks(arm_tag, x=move_xy[0], y=move_xy[1], max_step=max_step)

        cube_xy = cube.get_pose().p[:2]
        correction_xy = target_xy - cube_xy
        if np.linalg.norm(correction_xy) > 0.015:
            correction_xy = np.clip(correction_xy, -0.08, 0.08)
            self._move_by_chunks(arm_tag, x=correction_xy[0], y=correction_xy[1], max_step=min(max_step, 0.04))

        self._move_by_chunks(arm_tag, z=-lower_z, max_step=min(max_step, 0.04))
        self.move(self.open_gripper(arm_tag))
        self._move_by_chunks(arm_tag, z=lower_z, max_step=min(max_step, 0.04))

    def _place_cube_right_cross(self, cube, arm_tag, target_xy):
        self.move(self.grasp_actor(cube, arm_tag=arm_tag, pre_grasp_dis=0.08, grasp_dis=-0.015))
        if not self.plan_success:
            return

        self._move_by_chunks(arm_tag, z=0.22, max_step=0.035)
        if not self.plan_success:
            return

        target_high_pos = np.array(self.robot.get_right_ee_pose()[:3], dtype=float)
        cube_xy = cube.get_pose().p[:2]
        target_high_pos[:2] += target_xy - cube_xy

        high_path, low_path = self._plan_right_cross_place_paths(target_high_pos)
        if high_path is None:
            self.plan_success = False
            return

        self.right_joint_path.append(high_path)
        self._execute_arm_path(arm_tag, high_path)
        if low_path is not None:
            self.right_joint_path.append(low_path)
            self._execute_arm_path(arm_tag, low_path)

        self.move(self.open_gripper(arm_tag))
        self._move_by_chunks(arm_tag, z=0.08, max_step=0.04)

    def _plan_right_cross_place_paths(self, target_high_pos):
        current_quat = list(self.robot.get_right_ee_pose()[3:])
        candidate_quats = [
            current_quat,
            GRASP_DIRECTION_DIC["right_arm_perf"],
            GRASP_DIRECTION_DIC["top_down_little_right"],
            GRASP_DIRECTION_DIC["top_down"],
            GRASP_DIRECTION_DIC["down_right"],
            GRASP_DIRECTION_DIC["front_right"],
            GRASP_DIRECTION_DIC["front"],
            GRASP_DIRECTION_DIC["right"],
            GRASP_DIRECTION_DIC["down_left"],
            GRASP_DIRECTION_DIC["front_left"],
        ]
        candidate_poses = [list(target_high_pos) + list(quat) for quat in candidate_quats]
        high_paths = self.robot.right_plan_multi_path(candidate_poses)
        if "position" not in high_paths:
            return None, None

        lower_candidates = [0.10, 0.08, 0.05, 0.0]
        for pose_id, status in enumerate(high_paths["status"]):
            if status != "Success":
                continue
            high_path = {
                "status": "Success",
                "position": high_paths["position"][pose_id],
                "velocity": high_paths["velocity"][pose_id],
            }
            for lower_z in lower_candidates:
                if lower_z == 0.0:
                    return high_path, None
                low_pose = candidate_poses[pose_id].copy()
                low_pose[2] -= lower_z
                low_path = self.robot.right_plan_path(low_pose, last_qpos=high_path["position"][-1])
                if low_path["status"] == "Success":
                    return high_path, low_path
        return None, None

    def _execute_arm_path(self, arm_tag, path):
        save_freq = self.save_freq
        if save_freq is not None:
            self._take_picture()

        for step_id in range(path["position"].shape[0]):
            self.robot.set_arm_joints(path["position"][step_id], path["velocity"][step_id], str(arm_tag))
            self._step_scene()

            if self.render_freq and step_id % self.render_freq == 0:
                self._update_render()
                self.viewer.render()

            if save_freq is not None and step_id % save_freq == 0:
                self._update_render()
                self._take_picture()

        if save_freq is not None:
            self._take_picture()

    def _return_arm_to_home(self, arm_tag, steps=120):
        if arm_tag == "left":
            entity = self.robot.left_entity
            active_joints = self.robot.left_active_joints
            arm_joints_name = self.robot.left_arm_joints_name
            target_qpos = np.array(self.robot.left_homestate, dtype=float)
        else:
            entity = self.robot.right_entity
            active_joints = self.robot.right_active_joints
            arm_joints_name = self.robot.right_arm_joints_name
            target_qpos = np.array(self.robot.right_homestate, dtype=float)

        active_names = [joint.get_name() for joint in active_joints]
        arm_indices = [active_names.index(name) for name in arm_joints_name]
        current_qpos = np.array(entity.get_qpos(), dtype=float)[arm_indices]

        save_freq = self.save_freq
        if save_freq is not None:
            self._take_picture()

        for step in range(1, steps + 1):
            ratio = step / steps
            qpos = current_qpos * (1.0 - ratio) + target_qpos * ratio
            self.robot.set_arm_joints(qpos, np.zeros_like(qpos), str(arm_tag))
            self._step_scene()

            if self.render_freq and step % self.render_freq == 0:
                self._update_render()
                self.viewer.render()

            if save_freq is not None and step % save_freq == 0:
                self._update_render()
                self._take_picture()

        if save_freq is not None:
            self._update_render()
            self._take_picture()

    def _hold_final_pose(self, steps=120):
        save_freq = self.save_freq
        for step in range(1, steps + 1):
            self._step_scene()

            if self.render_freq and step % self.render_freq == 0:
                self._update_render()
                self.viewer.render()

            if save_freq is not None and step % save_freq == 0:
                self._update_render()
                self._take_picture()

        if save_freq is not None:
            self._update_render()
            self._take_picture()

    def _move_by_chunks(self, arm_tag, x=0.0, y=0.0, z=0.0, max_step=0.06):
        total = np.array([x, y, z], dtype=float)
        dist = np.linalg.norm(total)
        steps = max(1, int(np.ceil(dist / max_step)))
        delta = total / steps
        for _ in range(steps):
            self.move(
                self.move_by_displacement(
                    arm_tag,
                    x=float(delta[0]),
                    y=float(delta[1]),
                    z=float(delta[2]),
                    move_axis="world",
                )
            )

    def _in_box(self, actor):
        p = actor.get_pose().p
        cx, cy, _ = self.box_center
        hx, hy = self.box_half_xy
        min_center_z = self.box_floor_top_z + self.cube_half - self.success_height_tolerance
        max_center_z = self.box_wall_top_z - self.cube_half + self.success_height_tolerance
        return (
            cx - hx + self.cube_half < p[0] < cx + hx - self.cube_half
            and cy - hy + self.cube_half < p[1] < cy + hy - self.cube_half
            and min_center_z < p[2] < max_center_z
        )

    def _is_stable(self, actor):
        dynamic_component = next(
            (
                component
                for component in actor.actor.get_components()
                if isinstance(component, sapien.physx.PhysxRigidDynamicComponent)
            ),
            None,
        )
        if dynamic_component is None:
            return False
        return (
            np.linalg.norm(dynamic_component.get_linear_velocity())
            <= self.success_max_linear_velocity
            and np.linalg.norm(dynamic_component.get_angular_velocity())
            <= self.success_max_angular_velocity
        )

    def _success_conditions_met(self):
        return (
            self._in_box(self.right_cube)
            and self._in_box(self.left_cube)
            and self.is_right_gripper_open()
            and self.is_left_gripper_open()
            and self._is_stable(self.right_cube)
            and self._is_stable(self.left_cube)
        )

    def _step_scene(self):
        super()._step_scene()
        if not hasattr(self, "right_cube") or not hasattr(self, "left_cube"):
            return
        if self._success_conditions_met():
            self._success_stable_steps += 1
        else:
            self._success_stable_steps = 0

    def check_success(self):
        required_steps = max(1, round(self.success_hold_s / self.sim_timestep))
        return self._success_stable_steps >= required_steps
