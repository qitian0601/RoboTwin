# RoboTwin 项目完整工作报告

## 1. 项目概述

本项目位于：

```bash
/path/to/RoboTwin
```

该目录基于上游 **RoboTwin 2.0** 仓库进行本地扩展。RoboTwin 2.0 本身是一个面向双臂机器人操作的仿真数据生成与基准测试平台，提供大量任务、数字孪生资产、领域随机化能力、数据采集流程和多种策略基线支持。

本地工作的重点不是简单运行原始 RoboTwin demo，而是围绕 **NERO 双臂 embodiment** 和一个新的 **place_two_cubes_box 双臂放置任务** 进行适配、调试、数据采集和 LeRobot v3 格式转换。

当前项目已经完成的主要工作包括：

- 将 NERO 机械臂作为 RoboTwin embodiment 接入
- 新增 `place_two_cubes_box` 双臂任务
- 新增 NERO 双臂任务配置 `demo_nero_two_cubes`
- 修改采集脚本以支持 episode 续采和批量采集
- 增强 Base_Task，支持自定义桌面尺寸、桌面纹理和渲染质量
- 增强 Camera，支持手动腕部相机、D455/Large_D455 相机配置、RGB 畸变和图像拉伸
- 修改 HDF5/video 导出逻辑，使左右腕部相机视频也被保存
- 编写 RoboTwin HDF5 到 LeRobot v3 数据集的转换工具
- 编写采集完成后自动监控并转换数据的脚本
- 采集得到 `place_two_cubes_box/demo_nero_two_cubes` 数据集
- 生成对应的 LeRobot v3 数据集 `place_two_cubes_box_lerobot_v3`
- 增加一批 NERO 调试脚本，用于排查 URDF、IK、MotionGen、抓取、物理稳定性和任务执行

## 2. 原始 RoboTwin 平台能力

RoboTwin 2.0 是一个双臂机器人操作平台，主要能力包括：

- 双臂机器人仿真任务
- 大规模数据生成
- 强领域随机化
- 多种机器人 embodiment
- 多任务语言指令
- RGB、深度、点云、关节、末端位姿等多模态观测
- 任务成功判定与自动轨迹生成
- 多种策略基线支持

README 中列出的策略支持包括：

- DP
- ACT
- DP3
- RDT
- Pi0
- OpenVLA-oft
- TinyVLA
- DexVLA
- LLaVA-VLA
- GO-1

原始数据采集命令为：

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
```

例如：

```bash
bash collect_data.sh beat_block_hammer demo_randomized 0
```

RoboTwin 的采集逻辑通常分为两步：

1. 搜索可成功执行任务的随机 seed，并保存规划轨迹。
2. 根据成功 seed 重新播放轨迹，采集图像、状态、动作和语言数据。

## 3. 当前本地项目结构

项目根目录下的主要文件和目录如下：

```bash
RoboTwin/
├── README.md
├── readme.md
├── collect_data.sh
├── run_collect_200.sh
├── assets/
├── code_gen/
├── data/
├── description/
├── envs/
├── logs/
├── outputs/
├── policy/
├── script/
├── task_config/
└── tools/
```

其中，本地改动最集中的目录是：

- `assets/embodiments/nero/`
- `envs/place_two_cubes_box.py`
- `task_config/demo_nero_two_cubes.yml`
- `script/collect_data.py`
- `envs/_base_task.py`
- `envs/camera/camera.py`
- `envs/utils/pkl2hdf5.py`
- `tools/convert_robotwin_to_lerobot_v3.py`
- `tools/watch_collect_then_convert.sh`
- `run_collect_200.sh`

## 4. NERO embodiment 接入

### 4.1 新增 embodiment 配置

本地在 `task_config/_embodiment_config.yml` 中新增了 NERO：

```yaml
nero:
  file_path: "./assets/embodiments/nero"
```

这使 RoboTwin 的任务配置可以通过：

```yaml
embodiment: [nero, nero, 0.7]
```

加载左右两个 NERO 机械臂，并设置双臂间距。

### 4.2 NERO 资源文件

NERO 资源位于：

```bash
assets/embodiments/nero
```

主要内容包括：

```bash
config.yml
curobo.yml
curobo_tmp.yml
collision_nero.yml
nero_custom.yml
nero_with_gripper_description.urdf
nero_no_arm_collision.urdf
nero_no_arm_pad_collision.urdf
nero_step5_tmp.urdf
meshes/
```

mesh 资源包含：

- `base_link`
- `link1` 到 `link7`
- `gripper_base`
- `gripper_base_with_flange`
- `gripper_flange`
- `gripper_link1`
- `gripper_link2`
- `revo2_flange`

同时保留了 `.stl` 和 `.dae` 资源，供 URDF、Sapien 和渲染使用。

### 4.3 NERO config.yml

`assets/embodiments/nero/config.yml` 中定义了 NERO 的主要机器人配置：

- `urdf_path: nero_no_arm_collision.urdf`
- `planner: curobo`
- move group 为 `gripper_tcp`
- 7 个机械臂关节：`joint1` 到 `joint7`
- 夹爪主关节：`gripper_joint1`
- 夹爪 mimic：`gripper_joint2`
- `gripper_scale: [0.0, 0.05]`
- `gripper_bias: 0.08`
- 默认 homestate
- 坐标变换矩阵
- 机器人初始 pose
- joint stiffness / damping
- gripper stiffness / damping
- 抓取方向候选
- 默认静态相机配置

这部分工作使 NERO 能以 RoboTwin embodiment 的形式进入任务采集流程。

## 5. 新增任务：place_two_cubes_box

### 5.1 任务文件

新增任务文件为：

```bash
envs/place_two_cubes_box.py
```

该任务继承 `Base_Task`，定义了一个双臂协作任务：

```text
右臂抓取黄色方块并放入黑色盒子，随后左臂抓取绿色方块并放入同一个黑色盒子。
```

### 5.2 场景对象

任务中创建的主要对象包括：

- 黄色方块 `yellow_cube`
- 绿色方块 `green_cube`
- 黑色盒子底板 `black_box_floor`
- 黑色盒子四面墙

关键尺寸：

```text
cube_half = 0.025
box_length = 0.30
box_width = 0.20
box_height = 0.09
table_top_z = 0.74
```

方块初始位置会在左右两侧附近随机扰动：

```text
right cube: around [0.3, -0.3]
left cube:  around [-0.3, -0.3]
```

目标位置：

```text
right target xy = [0.10, box_center_y]
left target xy  = [-0.10, box_center_y]
```

### 5.3 任务执行逻辑

`play_once()` 中的执行顺序是：

1. 右臂抓取黄色方块。
2. 将黄色方块移动到黑盒右侧目标区域。
3. 右臂回到 home。
4. 左臂抓取绿色方块。
5. 将绿色方块移动到黑盒左侧目标区域。
6. 左臂回到 home。
7. 保持最终姿态一段时间，方便采集稳定画面。

核心方法包括：

- `_place_cube`
- `_move_by_chunks`
- `_return_arm_to_home`
- `_hold_final_pose`
- `_in_box`
- `check_success`

其中 `_move_by_chunks` 会把大位移切成多个小段，降低规划失败和物理不稳定的概率。

### 5.4 成功条件

任务成功条件是：

```python
return self._in_box(self.right_cube) and self._in_box(self.left_cube)
```

也就是两个方块都必须进入黑色盒子内部，并且高度高于桌面一定阈值。

### 5.5 语言指令模板

新增任务指令文件：

```bash
description/task_instruction/place_two_cubes_box.json
```

其中包含 seen 和 unseen 两类指令模板，例如：

```text
Use right arm to place the yellow cube into the black box, then use left arm to place the green cube into the black box.
```

该设计方便后续生成语言条件数据，用于 VLA 或 LeRobot 训练。

## 6. NERO 双臂任务配置

主要配置文件：

```bash
task_config/demo_nero_two_cubes.yml
```

该配置服务于 `place_two_cubes_box` 任务。

### 6.1 基本参数

主要参数包括：

```yaml
render_freq: 0
episode_num: 1
append_episode_on_run: true
use_seed: false
save_freq: 15
embodiment: [nero, nero, 0.7]
language_num: 1
save_path: ./data
clear_cache_freq: 1
collect_data: true
eval_video_log: true
```

其中 `append_episode_on_run: true` 是本地采集流程的重要改造，它允许每次运行时继续追加 episode，而不是覆盖旧数据。

### 6.2 双臂 pose 和 homestate override

配置中通过 `embodiment_overrides` 单独覆盖左右 NERO 的：

- homestate
- robot_pose

这部分用于将左右两个 NERO 放置到合适的双臂协作位置。

### 6.3 桌面配置

该任务使用更大的桌面：

```yaml
table_length: 1.4
table_width: 0.7
table_xy_bias: [0.0, -0.30]
table_texture_id: real_table_wood
```

这需要配合 `Base_Task` 的本地修改，使任务配置能够控制桌面尺寸和纹理。

### 6.4 相机配置

配置中使用三路 RGB 观测：

- front / head camera
- left wrist camera
- right wrist camera

相机类型包括：

```yaml
head_camera_type: Large_D455
wrist_camera_type: D455
```

front 相机配置：

```yaml
position: [0.0, -0.98, 1.52]
forward: [0.0, 0.72, -0.74]
left: [-1.0, 0.0, 0.0]
```

腕部相机使用手动配置：

```yaml
manual_wrist_camera: true
manual_wrist_camera_use_tcp_rotation: true
manual_wrist_camera_offset: [-0.16, 0.0, 0.12]
manual_wrist_camera_forward: [1.0, 0.0, -0.35]
manual_wrist_camera_left: [0.0, 1.0, 0.0]
```

此外，head camera 还配置了 RGB 畸变参数，用于模拟真实 RealSense 相机图像特性。

## 7. Base_Task 改造

修改文件：

```bash
envs/_base_task.py
```

主要改造包括：

### 7.1 setup_scene 支持配置参数

原始代码调用：

```python
self.setup_scene()
```

本地改为：

```python
self.setup_scene(**kwags)
```

这样渲染质量等参数可以从任务配置传入。

### 7.2 桌面尺寸和纹理可配置

`create_table_and_wall` 新增参数：

- `table_length`
- `table_width`
- `table_thickness`
- `table_texture_id`

这样不同任务可以使用不同桌面大小和材质。

### 7.3 渲染质量可配置

新增 `render_quality` 配置读取：

- `camera_shader_dir`
- `viewer_shader_dir`
- `ray_tracing_samples_per_pixel`
- `ray_tracing_path_depth`
- `ray_tracing_denoiser`

例如 `demo_nero_two_cubes.yml` 中使用：

```yaml
render_quality:
  camera_shader_dir: rt
  viewer_shader_dir: rt
  ray_tracing_samples_per_pixel: 256
  ray_tracing_path_depth: 8
  ray_tracing_denoiser: optix
```

### 7.4 wrist camera 更新时传入 TCP pose

本地修改 `_update_render`，将左右 TCP pose 传给相机系统：

```python
self.cameras.update_wrist_camera(
    self.robot.left_camera.get_pose(),
    self.robot.right_camera.get_pose(),
    self.robot.get_left_tcp_pose(),
    self.robot.get_right_tcp_pose(),
)
```

这为手动腕部相机绑定到 TCP 坐标系提供了基础。

## 8. Camera 系统改造

修改文件：

```bash
envs/camera/camera.py
```

Camera 改造是本项目的重要部分，目标是让采集图像更接近真实三相机机器人数据。

### 8.1 新增 D455 / Large_D455 相机配置

在 `task_config/_camera_config.yml` 中新增：

```yaml
D455:
  fovy: 63.74
  w: 640
  h: 400

Large_D455:
  fovy: 63.74
  w: 1280
  h: 800
```

### 8.2 静态相机可由任务配置覆盖

原本 static camera 默认来自 embodiment config。现在支持从任务配置中读取：

```yaml
camera:
  static_camera_list:
    - name: head_camera
      type: Large_D455
      position: [0.0, -0.98, 1.52]
      forward: [0.0, 0.72, -0.74]
      left: [-1.0, 0.0, 0.0]
```

这样可以为具体任务单独调相机位姿。

### 8.3 手动腕部相机

新增手动腕部相机能力：

- `manual_wrist_camera`
- `manual_wrist_camera_offset`
- `manual_wrist_camera_use_tcp_rotation`
- `manual_wrist_camera_forward`
- `manual_wrist_camera_left`
- `manual_wrist_camera_up`
- `manual_wrist_camera_look_at_offset`

如果启用 `manual_wrist_camera`，系统会根据 TCP pose 计算腕部相机 pose，而不是完全依赖机器人模型中已有相机 link。

### 8.4 RGB 畸变和图像拉伸

Camera 新增：

- `_apply_rgb_distortion`
- `_apply_rgb_stretch`

支持基于 inverse Brown-Conrady 风格参数进行图像畸变模拟，并支持 vertical/horizontal stretch。

配置示例：

```yaml
rgb_distortion:
  head_camera:
    enabled: true
    model: inverse_brown_conrady
    fx: 644.3301
    fy: 643.4210
    ppx: 642.9900
    ppy: 398.6344
    coeffs: [-0.0562166, 0.0662132, -0.0007664, 0.0004938, -0.0220463]
    vertical_stretch: 1.2
```

该功能用于让仿真图像更接近真实相机图像，利于后续 sim-to-real 或真实相机风格训练。

## 9. HDF5 与视频导出改造

修改文件：

```bash
envs/utils/pkl2hdf5.py
```

原始逻辑主要导出 head camera 视频。本地修改后，如果 observation 中存在左右腕部相机，也会额外导出：

```text
*_left_camera.mp4
*_right_camera.mp4
```

同时 HDF5 中仍保存完整 observation 数据。

这使 RoboTwin 原始数据可以保留三路视频观测：

- head_camera
- left_camera
- right_camera

## 10. 采集脚本改造

### 10.1 collect_data.py

修改文件：

```bash
script/collect_data.py
```

主要改造包括：

### 10.1.1 stderr 过滤

新增 `StderrFilter`，用于过滤 SAPIEN / Vulkan 的特定报错噪声：

```text
[svulkan2] [error] OIDN Error:
```

这样长时间批量采集日志更清晰。

### 10.1.2 embodiment overrides

新增 `apply_embodiment_overrides(args)`，允许任务配置覆盖左右机器人 config 中的字段，例如：

- homestate
- robot_pose
- gripper 参数

这对 NERO 双臂摆位非常重要。

### 10.1.3 append_episode_on_run

新增续采逻辑：

```yaml
append_episode_on_run: true
```

当该开关打开时，脚本会检查已有：

- `seed.txt`
- `data/episode*.hdf5`

并自动将 `episode_num` 调整到下一个需要采集的 episode。这样每次运行可以追加一个或多个 episode，而不会覆盖已有数据。

### 10.1.4 从已有 HDF5 继续采集

采集数据阶段会扫描已有 episode：

```python
while exist_hdf5(st_idx):
    st_idx += 1
```

然后从第一个不存在的 episode index 开始继续写入。

### 10.2 run_collect_200.sh

新增批量采集脚本：

```bash
run_collect_200.sh
```

默认参数：

```bash
RUNS=200
TASK_NAME=place_two_cubes_box
TASK_CONFIG=demo_nero_two_cubes
```

脚本逻辑：

1. 统计已有 HDF5 episode 数量。
2. 循环运行 `script/collect_data.py`。
3. 每次运行输出独立日志到 `logs/collect_*`。
4. 每次运行后再次统计 episode 数量。
5. 如果 episode 数量没有增加，则立即停止，避免重复失败。

典型运行方式：

```bash
./run_collect_200.sh
```

或指定次数：

```bash
./run_collect_200.sh 50
```

## 11. 数据采集结果

当前已经采集的主要数据集位于：

```bash
data/place_two_cubes_box/demo_nero_two_cubes
```

当前统计：

```text
HDF5 episodes: 53
instruction files: 53
trajectory pkl files: 53
```

主要内容包括：

```bash
data/place_two_cubes_box/demo_nero_two_cubes/data/episode*.hdf5
data/place_two_cubes_box/demo_nero_two_cubes/_traj_data/episode*.pkl
data/place_two_cubes_box/demo_nero_two_cubes/instructions/episode*.json
data/place_two_cubes_box/demo_nero_two_cubes/scene_info.json
```

其中：

- `_traj_data` 保存规划得到的左右关节路径
- `data` 保存最终 HDF5 采集数据
- `instructions` 保存每个 episode 对应的语言指令
- `scene_info.json` 保存每个 episode 的场景语义信息

## 12. RoboTwin 到 LeRobot v3 转换

### 12.1 转换脚本

新增转换工具：

```bash
tools/convert_robotwin_to_lerobot_v3.py
```

该脚本将 RoboTwin HDF5 数据转换为 LeRobot v3 格式。

输入结构：

```bash
data/episode0.hdf5
instructions/episode0.json
```

输出结构：

```bash
meta/info.json
meta/stats.json
meta/tasks.parquet
meta/episodes/chunk-000/file-000.parquet
data/chunk-000/file-000.parquet
videos/observation.images.*/chunk-000/file-*.mp4
```

### 12.2 state/action 设计

转换后的 state 和 action 均为 16 维，顺序为：

```text
right_nero_joint_1
right_nero_joint_2
right_nero_joint_3
right_nero_joint_4
right_nero_joint_5
right_nero_joint_6
right_nero_joint_7
left_nero_joint_1
left_nero_joint_2
left_nero_joint_3
left_nero_joint_4
left_nero_joint_5
left_nero_joint_6
left_nero_joint_7
right_gripper_width
left_gripper_width
```

RoboTwin 原始 HDF5 中的左右臂数据会被重排为：

```text
right arm 7D + left arm 7D + right gripper + left gripper
```

默认 action 模式是：

```text
observation.state[t] = qpos[t]
action[t] = qpos[t+1]
```

也就是 `next_state` 模式。这样 action 表示下一时刻的关节目标。

### 12.3 图像设计

转换后的 LeRobot 数据集包含三路视频：

```text
observation.images.front
observation.images.left_wrist
observation.images.right_wrist
```

默认尺寸：

```text
front:       800 x 1280 x 3
left_wrist:  480 x 640 x 3
right_wrist: 480 x 640 x 3
```

### 12.4 相机 metadata

转换脚本会向 `meta/info.json` 写入相机 metadata：

```text
front: intelrealsense, 1280x800, fps 30
left_wrist: intelrealsense, 640x480, fps 30
right_wrist: intelrealsense, 640x480, fps 30
```

并增加：

```json
"camera_features": {
  "front": "observation.images.front",
  "left_wrist": "observation.images.left_wrist",
  "right_wrist": "observation.images.right_wrist"
}
```

### 12.5 转换命令

当前使用命令记录在 `readme.md` 中：

```bash
conda run --no-capture-output -n lerobot python tools/convert_robotwin_to_lerobot_v3.py \
  --input-dir data/place_two_cubes_box/demo_nero_two_cubes \
  --output-dir data/place_two_cubes_box_lerobot_v3 \
  --repo-id place_two_cubes_box_lerobot_v3
```

## 13. 自动监控并转换工具

新增脚本：

```bash
tools/watch_collect_then_convert.sh
```

该脚本用于在批量采集时自动监控最新日志目录和 HDF5 输出，当采集稳定结束后自动启动 LeRobot 转换。

它会监控：

- HDF5 episode 数量
- run log 数量
- 最新日志修改时间
- `.cache` 目录状态
- 最新日志中是否出现 `Successfully Saved Instructions`

触发转换的条件包括：

- 连续若干轮状态稳定
- 或者已有 HDF5 且长时间无新活动，触发 partial convert

这适合长时间采集任务，避免人工等待采集结束后再手动转换。

## 14. LeRobot v3 数据集结果

转换后的数据集位于：

```bash
data/place_two_cubes_box_lerobot_v3
```

当前 `meta/info.json` 显示：

```text
robot_type: nero_dual
total_episodes: 53
total_frames: 19170
total_tasks: 2
fps: 30
observation.state shape: 16
action shape: 16
```

视频特征：

```text
observation.images.front:       800 x 1280 x 3
observation.images.left_wrist:  480 x 640 x 3
observation.images.right_wrist: 480 x 640 x 3
```

数据集目录包含：

```bash
data/place_two_cubes_box_lerobot_v3/data/chunk-000/file-000.parquet
data/place_two_cubes_box_lerobot_v3/meta/info.json
data/place_two_cubes_box_lerobot_v3/meta/stats.json
data/place_two_cubes_box_lerobot_v3/meta/tasks.parquet
data/place_two_cubes_box_lerobot_v3/meta/episodes/chunk-000/file-000.parquet
data/place_two_cubes_box_lerobot_v3/videos/observation.images.front/chunk-000/file-000.mp4
data/place_two_cubes_box_lerobot_v3/videos/observation.images.left_wrist/chunk-000/file-000.mp4
data/place_two_cubes_box_lerobot_v3/videos/observation.images.right_wrist/chunk-000/file-000.mp4
```

这说明本地已经完成从 RoboTwin 自动生成轨迹到 LeRobot v3 数据集的完整转换闭环。

## 15. 调试脚本体系

为了让 NERO embodiment 和新任务稳定运行，本地增加了多种调试脚本。

### 15.1 NERO embodiment 加载测试

文件：

```bash
script/test_nero_embodiment.py
```

用途：

- 直接用 Sapien 加载两个 NERO URDF
- 设置双臂 root pose
- 设置 homestate
- 设置夹爪开合
- step 若干帧
- 打印 active joints 和 links
- 验证 NERO 双臂模型能否正确加载

### 15.2 NERO 抓取调试

文件：

```bash
script/debug_nero_grasp.py
```

用途：

- 调试 NERO 抓取姿态
- 检查候选 grasp pose
- 调用 cuRobo IK
- 检查不同旋转候选是否可解
- 打印 planner base target
- 分析抓取失败原因

### 15.3 NERO MotionGen 调试

文件：

```bash
script/debug_nero_motiongen.py
```

用途：

- 构造 cuRobo MotionGen
- 测试 IK 和 motion generation
- 检查 base 坐标系下目标位姿
- 调试 self-collision、world collision 和 trajectory optimization

### 15.4 NERO 物理稳定性调试

文件：

```bash
script/debug_nero_physics_stability.py
```

用途：

- 加载不同 collision 变体 URDF
- 测试双臂初始状态是否稳定
- 检查 joint 是否越界
- 检查左右臂之间或自身 link 的接触
- 比较 original / no_body / no_arm / no_all 等碰撞模式

### 15.5 NERO full task 调试

文件：

```bash
script/debug_nero_full_task.py
```

用途：

- 在已有任务上完整跑 NERO 抓取和放置流程
- 打印每一步规划状态
- 检查 gripper contact
- 检查任务成功条件

### 15.6 place_two_cubes_box 调试

文件：

```bash
script/debug_place_two_cubes_box.py
```

用途：

- 加载 `place_two_cubes_box` 任务
- 打印方块、目标和末端位置
- 执行完整 `play_once`
- 打印左右关节路径变化
- 检查 `check_success`

这些脚本说明本地工作不是只写任务文件，而是围绕机器人模型、规划器、碰撞、抓取、相机和数据采集进行了系统性调试。

## 16. 渲染与材质测试

输出目录中存在：

```bash
outputs/sapien_rt_demo/material_raster.png
outputs/sapien_rt_demo/material_rt_256spp.png
```

这说明项目中还进行了 Sapien raster 和 ray tracing 材质渲染对比测试。相关脚本包括：

```bash
script/rt_material_demo.py
```

该工作与后续高质量图像采集、相机真实感和视觉策略训练有关。

## 17. 当前工作区状态

当前 RoboTwin 工作区有未提交修改和新增文件。

### 17.1 已修改文件

```text
envs/_base_task.py
envs/beat_block_hammer.py
envs/camera/camera.py
envs/utils/pkl2hdf5.py
script/_install.sh
script/collect_data.py
task_config/_camera_config.yml
task_config/_embodiment_config.yml
```

这些修改主要涉及：

- NERO embodiment 支持
- 桌面和渲染配置增强
- 手动腕部相机和 RGB 畸变
- HDF5/video 导出增强
- 数据采集续采逻辑
- 安装脚本 pip 调用方式修正

### 17.2 新增文件和目录

```text
description/task_instruction/place_two_cubes_box.json
envs/place_two_cubes_box.py
logs/
outputs/
readme.md
run_collect_200.sh
script/debug_nero_full_task.py
script/debug_nero_grasp.py
script/debug_nero_motiongen.py
script/debug_nero_physics_stability.py
script/debug_nero_step5.py
script/debug_nero_step6.py
script/debug_place_two_cubes_box.py
script/debug_place_two_cubes_reach.py
script/make_nero_step5_urdf.py
script/rt_material_demo.py
script/test_nero_embodiment.py
tools/
```

这些新增内容集中体现了 NERO 双臂任务接入和数据转换流水线。

## 18. 本地操作命令记录

`readme.md` 中记录了当前常用命令。

### 18.1 NERO smoke 数据采集

```bash
python script/collect_data.py beat_block_hammer demo_nero_smoke
```

### 18.2 NERO full task debug

```bash
python -u script/debug_nero_full_task.py --seeds 1 --render-freq 1
```

### 18.3 place_two_cubes_box 单次采集

```bash
cd RoboTwin
conda activate RoboTwin5090
conda run --no-capture-output -n RoboTwin5090 \
  python -u script/collect_data.py place_two_cubes_box demo_nero_two_cubes
```

### 18.4 转换 LeRobot v3

```bash
conda run --no-capture-output -n lerobot python tools/convert_robotwin_to_lerobot_v3.py \
  --input-dir data/place_two_cubes_box/demo_nero_two_cubes \
  --output-dir data/place_two_cubes_box_lerobot_v3 \
  --repo-id place_two_cubes_box_lerobot_v3
```

### 18.5 批量采集

```bash
./run_collect_200.sh
```

## 19. 技术亮点

### 19.1 将 NERO 接入 RoboTwin embodiment 系统

本地工作成功将 NERO 机械臂配置成 RoboTwin 可识别的 embodiment，包含 URDF、mesh、gripper、homestate、planner 和相机配置。

### 19.2 自定义双臂协作任务

新增 `place_two_cubes_box` 任务，包含完整场景构建、对象随机化、双臂顺序操作、路径执行和成功判定。

### 19.3 高质量三相机数据采集

通过手动腕部相机、Large_D455 / D455 配置、相机畸变和图像拉伸，数据采集更贴近真实机器人三相机设置。

### 19.4 自动续采和批量采集

`append_episode_on_run` 和 `run_collect_200.sh` 解决了长时间数据采集时最常见的问题：重复覆盖、失败重跑和日志混乱。

### 19.5 RoboTwin 到 LeRobot v3 的数据桥接

`convert_robotwin_to_lerobot_v3.py` 将 RoboTwin 生成的 HDF5 数据转换为 LeRobot v3 标准结构，使数据可以直接用于 LeRobot 生态中的策略训练。

### 19.6 系统性调试脚本

围绕 NERO 的加载、碰撞、抓取、IK、MotionGen 和任务执行建立了多个调试入口，降低后续迭代成本。

## 20. 当前问题与风险

### 20.1 当前工作区未提交

当前有较多未提交修改和新增文件，建议在确认稳定后进行一次整理提交，避免本地实验成果丢失。

### 20.2 数据规模仍较小

当前 LeRobot v3 数据集为：

```text
53 episodes, 19170 frames
```

如果目标是训练稳定策略，后续可能需要扩展到数百或上千条 episode。

### 20.3 success rate 还缺少统一统计

目前日志中有多次采集记录，但报告级别还没有统一统计：

- seed 搜索成功率
- 每次采集失败原因
- 平均 episode 时长
- 平均轨迹步数
- 转换后数据完整性检查

建议后续补充自动评估脚本。

### 20.4 action 目前是 next qpos

转换后的 action 是下一帧 qpos，而不是末端位姿或显式控制命令。这对 ACT / Diffusion 类型策略是可用的，但后续如果要接 Pi0/Pi0.5 或真实机器人，可能需要进一步确认动作语义是否匹配目标策略接口。

### 20.5 wrist camera 几何需要视觉验证

手动腕部相机虽然已经实现，但仍建议系统检查：

- 相机是否稳定跟随 TCP
- 左右腕相机是否朝向合理
- 图像是否被机械臂自身严重遮挡
- 畸变和拉伸是否过强

## 21. 后续建议

### 21.1 增加数据完整性检查脚本

建议新增脚本检查：

- HDF5 episode 数量
- instruction 数量
- pkl 轨迹数量
- 每个 HDF5 的帧数
- 三路相机是否存在
- qpos/action 是否有 NaN
- LeRobot parquet 和 video 是否一致

### 21.2 增加批量评估统计

建议从 logs 中自动提取：

- 总运行次数
- 成功次数
- 失败次数
- 常见失败原因
- 每次新增 episode 数量

### 21.3 固化 LeRobot schema

建议明确当前标准 schema：

```text
robot_type: nero_dual
fps: 30
observation.state: 16D qpos/gripper
action: 16D next qpos/gripper
images: front + left_wrist + right_wrist
```

并在 README 或专门文档中说明。

### 21.4 扩展数据规模

继续使用：

```bash
./run_collect_200.sh
```

将数据扩展到更大规模，同时用 `watch_collect_then_convert.sh` 自动转换。

### 21.5 对接策略训练

下一步可以使用转换后的 LeRobot v3 数据集训练：

- ACT
- Diffusion Policy
- Pi0 / Pi0.5
- OpenVLA-oft

训练前需要确认策略接口是否接受 16D qpos action 和三路视频输入。

## 22. 总结

本地 RoboTwin 项目的主要成果是：在原始 RoboTwin 2.0 平台基础上，完成了 **NERO 双臂 embodiment 接入、双方块入盒任务构建、三相机高质量采集、批量采集流程、HDF5 到 LeRobot v3 转换** 的完整工程闭环。

当前最核心的产物包括：

1. `assets/embodiments/nero/`：NERO embodiment 资源和配置。
2. `envs/place_two_cubes_box.py`：新增双臂放置任务。
3. `task_config/demo_nero_two_cubes.yml`：NERO 双臂任务配置。
4. `run_collect_200.sh`：批量采集脚本。
5. `tools/convert_robotwin_to_lerobot_v3.py`：LeRobot v3 转换工具。
6. `data/place_two_cubes_box/demo_nero_two_cubes`：RoboTwin 原始采集数据，53 episodes。
7. `data/place_two_cubes_box_lerobot_v3`：转换后的 LeRobot v3 数据集，53 episodes、19170 frames、三路视频、16D state/action。

整体来看，该项目已经具备从 RoboTwin 仿真任务自动生成 NERO 双臂演示数据，并转换到 LeRobot 训练格式的能力。后续重点应放在扩大数据规模、补充自动质量检查、量化采集成功率，以及对接具体策略训练上。
