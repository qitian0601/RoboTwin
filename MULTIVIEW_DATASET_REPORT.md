# RoboTwin `place_two_cubes_box` 多视角配对数据报告

## 1. 目标

本数据集通过回放已有 53 条成功专家轨迹中的 50 条，在同一个 simulator state 上同步渲染多个相机视角。策略动作始终来自原始 `_traj_data/episode*.pkl`，不会针对 C1/C2/C3/C4 分别推理或重新规划，因此各视角是严格的状态级配对数据。

输出目录：

```text
data/place_two_cubes_box_multiview_50
```

## 2. 数据选择与划分

- 总数：50 条轨迹。
- 两种语义等价指令：各 25 条，避免语言分布失衡。
- `train`：40 条，每条保存 C0/C1/C2 和双腕相机。
- `val`：5 条，每条保存 C0/C1/C2 和双腕相机。
- `test`：5 条，额外保存未参与训练的 C3/C4。
- 保留源轨迹：46、48、49，不写入当前 50 条数据集。
- 完整映射见 `manifest.json`，不要根据目录编号反推源 episode。

## 3. 相机定义

所有中间相机位置、内参、镜头畸变参数与 C0 相同，只绕相机局部 left 轴改变俯仰：

| 名称 | 俯仰变化 | 用途 |
| --- | ---: | --- |
| C0 | 0° | canonical reference |
| C1 | +15° | Adapter 训练/验证 |
| C2 | -15° | Adapter 训练/验证 |
| C3 | +25° | 未见视角测试 |
| C4 | -25° | 未见视角测试 |
| left_wrist | 随左 TCP 更新 | 训练/验证/测试 |
| right_wrist | 随右 TCP 更新 | 训练/验证/测试 |

一次采样只调用一次 `scene.update_render()`，之后所有相机连续 `take_picture()`，中间没有 `scene.step()`。因此同一 frame index 的 C0-C4、双腕图像、机器人状态和物体状态对应同一个仿真时刻。

## 4. 采样与数值约定

- 仿真频率：250 Hz。
- 数据频率：10 Hz，即每 25 个 physics steps 采样一次。
- 单条最长记录 20 秒；任务提前完成时保留完整实际时长，不补重复帧。
- `timestamp` 从 0 开始，严格以 0.1 秒递增。
- 夹爪统一使用米，范围为 `0-0.1 m`。
- `observation/state` 和 `action/commanded` 均为 16 维。
- 16 维顺序：右臂 7、左臂 7、右夹爪宽度、左夹爪宽度。
- `observation/state` 使用实际 qpos 和实际夹爪宽度。
- `action/commanded` 使用当前 drive target 和命令夹爪宽度。

## 5. 单条轨迹结构

```text
train/episode_000/
├── _SUCCESS
├── c0.mp4
├── c1.mp4
├── c2.mp4
├── left_wrist.mp4
├── right_wrist.mp4
├── data.hdf5
├── episode.json
└── instruction.json
```

测试条目另外包含 `c3.mp4` 和 `c4.mp4`。

`_SUCCESS` 只在 HDF5、所有视频封装及逐帧校验全部通过后生成。缺少该文件的目录是不完整中间结果，续跑时会自动删除并重做。

## 6. HDF5 主要字段

```text
/timestamp                              (N,)
/sim_step                               (N,)
/frame_index                            (N,)
/success                                (N,)
/observation/state                      (N, 16)
/action/commanded                       (N, 16)
/robot/{left,right}/qpos                (N, 7)
/robot/{left,right}/qvel                (N, 7)
/robot/{left,right}/commanded_qpos      (N, 7)
/robot/{left,right}/ee_pose             (N, 7)
/robot/{left,right}/gripper_width_m     (N,)
/objects/{yellow_cube,green_cube}/pose  (N, 7)
/objects/.../linear_velocity            (N, 3)
/objects/.../angular_velocity           (N, 3)
/cameras/{view}/intrinsic_cv            (N, 3, 3)
/cameras/{view}/extrinsic_cv            (N, 3, 4)
/cameras/{view}/cam2world_gl            (N, 4, 4)
```

每个 HDF5 还保存任务、源 episode、split、频率、夹爪单位、状态顺序、最终成功标记和相机俯仰配置。

## 7. 运行与恢复

全量运行：

```bash
cd /path/to/RoboTwin
./run_replay_multiview_50.sh
```

指定源 episode：

```bash
./run_replay_multiview_50.sh --episodes 0,25,52
```

只校验已完成条目：

```bash
./run_replay_multiview_50.sh --validate-only
```

脚本默认断点续跑：已有 `_SUCCESS` 的条目先校验再跳过；不完整条目自动重做。只有显式传入 `--overwrite` 才会覆盖已完成条目。

## 8. 已验证事项

- 新增相机不消耗任务随机数，方块初始状态与原专家路径保持一致。
- 烟雾测试覆盖源 episode 0、25、52。
- 训练视角 5 路视频和测试视角 7 路视频均通过逐帧计数。
- HDF5 时间戳间隔严格为 0.1 秒。
- 实际/命令夹爪值均使用 `0-0.1 m`。
- 三条烟雾测试最终任务成功，且相机画面非空。

## 9. 使用边界

- C3/C4 是更强的未见视角，可能出现明显裁切，只用于测试，不加入 Adapter 训练。
- 这是专家轨迹回放数据，不是 π0.5 闭环采集数据。
- 原始 53 条 HDF5 和 `_traj_data` 均保持不变。
- 当前输出直接适合按 frame index 构造 C1/C0、C2/C0 图像对；转 LeRobot 前仍应明确 action 是当前 drive target 还是 next-state action。

## 10. 视角多样性续集

首批生成在 19 条时暂停并完整保留。其余 31 条使用较温和的训练俯仰继续采集：

- 首批目录：`data/place_two_cubes_box_multiview_50`，C1/C2 为 `±15°`，C3/C4 为 `±25°`。
- 续集目录：`data/place_two_cubes_box_multiview_50_pitch10_continuation`，C1/C2 为 `±10°`，C3/C4 为 `±15°`。
- 续集 manifest 自动排除首批所有带 `_SUCCESS` 的源 episode，因此两批合计仍是 50 条，不会重复训练轨迹。
- 联合训练时可把 `±10°` 和 `±15°` 视为两档 camera-shift augmentation；评估结果应按 pitch 档位分别汇报。
