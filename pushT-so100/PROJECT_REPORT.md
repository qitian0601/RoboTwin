# pushT-so100 项目完整工作报告

## 1. 项目概述

`pushT-so100` 项目最初围绕 **SO100 机械臂在 MuJoCo 仿真环境中完成 PushT 操作任务**展开，目标是建立一套可以完成数据采集、数据转换、策略训练和策略推理的机器人模仿学习流程。随着项目推进，系统从早期的 2D PushT / SO100 任务逐步扩展到 **NERO 7DoF 机械臂 + 夹爪抓取任务**，形成了一个更完整的机器人学习实验平台。

当前项目已经覆盖以下核心能力：

- MuJoCo 机器人仿真环境搭建
- SO100 / NERO 机械臂模型集成
- 手柄遥操作数据采集
- LeRobot 格式数据集生成与维护
- Diffusion Policy 策略训练与推理
- ACT 策略训练与推理
- Pi0.5 策略训练与推理适配
- 训练 checkpoint、TensorBoard 日志和推理视频保存
- 数据集 episode 删除、索引修复、统计量重算
- TCP / mocap 坐标系转换与状态格式升级

总体来看，本项目已经从一个单任务仿真 demo 发展成了一个具有完整数据闭环和多策略实验能力的机器人模仿学习管线。

## 2. 项目目标

项目目标可以分为三个阶段。

### 2.1 初始目标：SO100 PushT 任务

早期目标是在 MuJoCo 中构建 SO100 机械臂操作环境，使机械臂能够通过末端控制完成 PushT 类任务，并使用人类遥操作数据训练模仿学习策略。

这一阶段的主要工作包括：

- 构建 SO100 MuJoCo 场景
- 定义 PushT 目标物体和桌面环境
- 实现 Gymnasium 兼容环境
- 实现遥操作采集脚本
- 将数据保存为 LeRobot 数据集格式
- 使用 Diffusion Policy 进行训练和推理

### 2.2 扩展目标：完整 LeRobot 工作流

随后项目重点扩展为完整的 LeRobot 工作流，目标是让采集、训练、推理都能围绕统一的数据格式运行。

这一阶段完成了：

- LeRobotDataset 数据写入
- 数据集 metadata 和 stats 管理
- 双相机图像观测
- 策略训练脚本参数化
- 策略推理脚本参数化
- TensorBoard 日志记录
- checkpoint 定期保存
- 推理视频录制

### 2.3 当前目标：NERO 7DoF 抓取任务

后期项目明显从原始 PushT 任务扩展到 NERO 7 自由度机械臂抓取任务。当前主线任务更接近：机械臂通过夹爪抓取桌面上的红色圆柱体，并将目标物体抬起。

这一阶段的关键变化是：

- 引入 NERO 7DoF 机械臂模型
- 引入夹爪模型和夹爪 actuator
- 将 action 从早期低维控制升级为 8 维 TCP action
- 将 observation.state 从关节状态升级为关节 + TCP 状态
- 增加 TCP 与 mocap 坐标转换模块
- 支持 Diffusion、ACT、Pi0.5 三类策略实验
- 对 MuJoCo 接触、夹爪控制和目标物体物理参数进行调试

## 3. 项目目录结构

项目根目录位于：

```bash
/home/qt/Downloads/pushT-so100
```

主要目录和文件如下：

```bash
pushT-so100/
├── README.md
├── README-ZH.md
├── readme.md
├── environment.yml
├── delete_episodes.py
├── test.py
├── script/
│   ├── record_demonstration_data.sh
│   ├── train_policy.sh
│   └── infer.sh
├── src/
│   ├── env_human_ee.py
│   ├── env_gym_ee.py
│   ├── train_diffusion.py
│   ├── train.py
│   ├── infer.py
│   ├── infer_act.py
│   ├── infer_pi.py
│   ├── state_features.py
│   ├── tcp_frame.py
│   ├── augment_observation_state.py
│   ├── inspect_mocap_tcp_offset.py
│   ├── helper.py
│   └── fix.py
├── data/
│   ├── NewData3.9-ee-2d-pos/
│   ├── nero-dataset/
│   └── nero-dataset_backup/
├── outputs/
│   ├── nero_diffusion/
│   ├── nero_diffusion_v2/
│   ├── nero_diffusion_official/
│   ├── nero_act/
│   ├── nero_act_v2/
│   ├── nero_act_v3/
│   ├── nero_act_v4/
│   ├── recorded_videos/
│   └── recorded_videos_pi05/
├── assets/
└── chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/
    ├── so_arm100.xml
    ├── scene.xml
    ├── human_env.xml
    ├── nero_arm.xml
    ├── nero_human_env.xml
    ├── nero_scene.xml
    ├── assets/
    └── nero/
```

其中，`src/` 是核心源码目录，`data/` 保存 LeRobot 数据集，`outputs/` 保存训练和推理产物，`chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/` 保存 MuJoCo 模型、场景、mesh 和 URDF 资源。

## 4. 仿真环境建设

### 4.1 SO100 环境

项目早期基于 SO100 机械臂构建 PushT 任务环境。相关模型和场景文件包括：

- `so_arm100.xml`
- `so_arm100_leader.xml`
- `scene.xml`
- `human_env.xml`
- `test_env.xml`

早期任务的主要特点是：

- 机械臂在桌面环境中操作目标物体
- 使用 MuJoCo 进行物理仿真
- 使用相机渲染图像观测
- 使用 Gymnasium 接口封装训练环境
- 早期 action 维度较低，主要面向平面控制

### 4.2 NERO 机械臂环境

后期项目引入 NERO 7DoF 机械臂和夹爪模型。相关资源包括：

- `nero_arm.xml`
- `nero_human_env.xml`
- `nero_scene.xml`
- `nero/meshes/*.stl`
- `nero/meshes/dae/*.dae`
- `nero/urdf/*.urdf`
- `nero/urdf/*.xacro`

NERO 环境的主要特点是：

- 7 个机械臂关节
- 双指夹爪
- TCP site：`gripper_tcp`
- mocap body：`target_mocap`
- 目标物体：红色圆柱体 `target_block`
- 双相机视角：`top_view` 和 `side_view`
- 桌面接触和夹爪接触经过调参

### 4.3 物理参数调试

为了提高抓取稳定性和减少仿真异常，项目对 MuJoCo XML 中的物理参数进行了多轮调整。

主要调整包括：

- 将部分 geom 的 `condim` 调整为 6，提高接触约束表达能力
- 增加夹爪 slide joint 的 damping、armature 和 frictionloss
- 增强夹爪 actuator 的 `kp` 和 `kv`
- 调整桌面 geom 的 `solimp`、`solref`、`friction` 和 `condim`
- 调整目标圆柱体质量和摩擦参数

这些修改的目的是让夹爪闭合、接触和物体抬升更稳定，减少目标物体穿透、抖动或无法夹持的问题。

## 5. 遥操作数据采集

数据采集脚本主要是：

```bash
src/env_human_ee.py
```

该脚本通过游戏手柄控制机械臂末端目标，并将采集到的图像、状态、动作保存为 LeRobot 数据集。

### 5.1 输入设备

脚本使用 `pygame` 读取手柄输入，支持 Xbox / PS 类控制器。手柄输入被映射到：

- TCP / mocap 位置移动
- TCP / mocap 姿态旋转
- 夹爪开合
- 开始录制
- 停止录制并保存 episode
- 环境 reset

如果未检测到手柄，脚本会进入 keyboard-only 模式，但实际控制主要依赖手柄。

### 5.2 录制内容

每一帧录制的数据包括：

- `observation.images.cam_top`
- `observation.images.cam_side`
- `observation.state`
- `action`
- `task`

当前 NERO 数据格式中，`observation.state` 是 15 维：

```text
joint1, joint2, joint3, joint4, joint5, joint6, joint7,
tcp_x, tcp_y, tcp_z, tcp_qw, tcp_qx, tcp_qy, tcp_qz, gripper
```

`action` 是 8 维：

```text
tcp_x, tcp_y, tcp_z, tcp_qw, tcp_qx, tcp_qy, tcp_qz, gripper
```

### 5.3 数据保存方式

数据通过 `LeRobotDataset` 写入本地数据集目录。脚本支持：

- 如果数据集存在，则继续追加 episode
- 如果数据集不存在，则创建新的 LeRobot dataset
- 录制结束后异步保存，避免主循环阻塞
- 对连续重复 action 做去重，减少无效帧

典型采集命令如下：

```bash
python src/env_human_ee.py \
  --repo_id ./data/nero-dataset \
  --fps 10 \
  --move_speed 0.05 \
  --rot_speed 0.3
```

## 6. Gymnasium 环境封装

训练和推理环境主要由以下文件实现：

```bash
src/env_gym_ee.py
```

该文件定义了 `PushT` 环境类，虽然类名仍保留 PushT，但当前实际功能已经扩展到 NERO 抓取任务。

### 6.1 action space

当前 action space 是 8 维 Box：

```text
[x, y, z, qw, qx, qy, qz, gripper]
```

其中：

- `x, y, z` 是 TCP 目标位置
- `qw, qx, qy, qz` 是 TCP 目标姿态四元数
- `gripper` 是夹爪开合状态，0 表示打开，1 表示闭合

### 6.2 observation space

当前 observation 包括：

```text
cam_top:  224 x 224 x 3 RGB 图像
cam_side: 224 x 224 x 3 RGB 图像
observation.state: 15 维 float32 状态
```

其中 `observation.state` 包含 7 个关节位置和 8 维 TCP / gripper 状态。

### 6.3 step 逻辑

环境 step 的主要流程是：

1. 接收策略输出的 8D TCP action
2. 对 TCP 位置做工作空间限幅
3. 对四元数归一化
4. 将 TCP pose 转换成 mocap pose
5. 使用位置插值和 SLERP 平滑移动 mocap
6. 根据 action 控制夹爪开合
7. 执行多个 MuJoCo simulation step
8. 渲染双相机图像
9. 读取关节状态和 TCP 状态
10. 计算 reward、terminated、truncated 和 info

### 6.4 reward 和成功条件

当前 reward 面向抓取任务设计：

- 基础奖励：末端 TCP 和目标物体距离的负值
- 额外奖励：夹爪闭合且靠近目标物体
- 成功奖励：目标物体被抬起

当前成功条件是：

```text
target_block 的 z 坐标 > 0.45
```

这表示目标圆柱体已经被机械臂抓起或抬离桌面。

## 7. TCP 与 mocap 坐标转换

项目新增了专门的坐标转换模块：

```bash
src/tcp_frame.py
```

该模块解决了一个关键问题：MuJoCo 中用于控制的 mocap body 与夹爪真实 TCP site 不完全重合。如果直接把策略 action 当作 mocap pose 使用，控制目标会和真实末端位置存在固定偏差。

因此项目实现了以下工具函数：

- MuJoCo `wxyz` 四元数与 scipy `xyzw` 四元数互转
- 四元数归一化
- pose 转 4x4 齐次矩阵
- 4x4 齐次矩阵转 pose
- 读取 body pose
- 读取 site pose
- 计算 mocap 到 TCP 的固定变换
- mocap pose 转 TCP pose
- TCP pose 转 mocap pose
- 打包 `[pos, quat, gripper]` 为统一状态

这部分工作使数据采集、环境 step、旧数据转换和策略推理能够使用统一的 TCP 表示。

## 8. 数据集建设

项目中保留了两个主要数据集阶段。

### 8.1 早期 SO100 数据集

路径：

```bash
data/NewData3.9-ee-2d-pos
```

该数据集信息如下：

```text
robot_type: so100_arm
total_episodes: 208
total_frames: 25923
fps: 10
observation.state shape: 5
action shape: 2
```

它对应早期 SO100 / PushT 风格任务，状态和动作维度较低。

### 8.2 当前 NERO 数据集

路径：

```bash
data/nero-dataset
```

该数据集信息如下：

```text
robot_type: nero_arm
total_episodes: 150
total_frames: 28666
total_tasks: 1
fps: 10
observation.images.cam_top: 224 x 224 x 3 video
observation.images.cam_side: 224 x 224 x 3 video
observation.state shape: 15
action shape: 8
```

该数据集是当前 Diffusion、ACT、Pi0.5 实验的主要数据来源。

## 9. 数据集维护工具

### 9.1 删除 episode 工具

文件：

```bash
delete_episodes.py
```

该脚本用于删除数据集中质量较差、损坏或不需要的 episode。它不仅删除文件，还会修复 LeRobot 数据集内部的索引和 metadata。

支持的能力包括：

- 支持 `0-9,55-59` 范围格式
- 支持 `3,7,12` 离散 episode 格式
- 支持 dry-run 预览
- 自动备份数据集
- 删除 data parquet
- 删除对应视频文件
- 删除 meta episode 文件
- 重排 episode 编号
- 修复 `episode_index`
- 修复全局 `index`
- 修复 `dataset_from_index` 和 `dataset_to_index`
- 修复视频 file index
- 更新 `meta/info.json`
- 重新计算 `meta/stats.json`
- 清理 Hugging Face / LeRobot cache

典型命令：

```bash
# 预览删除
python delete_episodes.py --episodes 0-9,55-59 --dry-run

# 正式删除
python delete_episodes.py --episodes 44

# 指定数据集路径
python delete_episodes.py --episodes 5,10,15 --data-dir ./data/nero-dataset

# 跳过备份
python delete_episodes.py --episodes 0-4 --no-backup
```

### 9.2 observation.state 增强工具

文件：

```bash
src/augment_observation_state.py
```

该脚本用于处理旧数据格式和新数据格式之间的迁移。

支持模式包括：

- `prev_action`：用上一帧 action 作为当前 EE state
- `same_action`：用当前帧 action 作为 EE state
- `mujoco_tcp`：基于 MuJoCo 正运动学和坐标变换，将旧 mocap action 转成 TCP action / state
- `revert_7d`：恢复到仅关节状态

该工具会同步更新：

- parquet 数据文件
- `meta/info.json`
- `meta/stats.json`
- action 统计量
- observation.state 统计量

这部分是项目从旧的 mocap 控制格式迁移到 TCP 控制格式的重要工具。

## 10. Diffusion Policy 训练

Diffusion Policy 训练脚本为：

```bash
src/train_diffusion.py
```

该脚本基于 LeRobot 的 `DiffusionPolicy` 实现训练。

### 10.1 训练流程

训练流程如下：

1. 读取命令行参数
2. 创建输出目录和 checkpoint 目录
3. 读取 LeRobotDatasetMetadata
4. 将数据集 features 转换为 policy features
5. 设置图像输入 shape 为 `(3, 224, 224)`
6. 构造 input features 和 output features
7. 创建 `DiffusionConfig`
8. 构造 delta timestamps
9. 加载 LeRobotDataset
10. 创建 DiffusionPolicy
11. 创建 preprocessor 和 postprocessor
12. 创建 optimizer 和 learning rate scheduler
13. 使用 DataLoader 训练
14. 记录 TensorBoard loss 和 lr
15. 定期保存 checkpoint
16. 保存 final_model

### 10.2 主要训练参数

常用参数包括：

```text
batch-size
training-steps
warmup-steps
log-freq
save-freq
lr
weight-decay
num-workers
n-obs-steps
horizon
n-action-steps
vision-backbone
device
image-keys
mask-keys
```

典型训练命令：

```bash
python src/train_diffusion.py \
  --data-path ./data/nero-dataset \
  --output-dir outputs/nero_diffusion_v2 \
  --batch-size 16 \
  --training-steps 15000 \
  --warmup-steps 1000 \
  --log-freq 50 \
  --save-freq 2000 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --num-workers 8 \
  --n-obs-steps 2 \
  --horizon 16 \
  --n-action-steps 8 \
  --vision-backbone resnet18 \
  --device cuda
```

### 10.3 Diffusion 实验产物

项目中已有多个 Diffusion 实验目录：

```bash
outputs/nero_diffusion
outputs/nero_diffusion_v2
outputs/nero_diffusion_official
```

其中 `nero_diffusion_official` 包含从 step 5000 到 step 95000 的多个 checkpoint，说明已经进行了较长训练和多轮实验。

## 11. ACT 策略训练

ACT 训练脚本为：

```bash
src/train.py
```

该脚本基于 LeRobot 的 `ACTPolicy` 实现。

### 11.1 训练流程

ACT 训练流程包括：

1. 读取数据集 metadata
2. 构造 policy features
3. 创建 `ACTConfig`
4. 使用 `resolve_delta_timestamps` 自动解析时间窗口
5. 加载 LeRobotDataset
6. 创建 ACTPolicy
7. 创建 preprocessor / postprocessor
8. 将 backbone 参数和其他参数分组
9. 使用 AdamW 优化器
10. 使用 warmup + cosine decay scheduler
11. 训练过程中记录 loss、L1 loss、KLD loss 和 learning rate
12. 定期保存 checkpoint
13. 保存 final_model

### 11.2 ACT 训练参数

主要参数包括：

```text
batch-size
training-steps
warmup-steps
log-freq
save-freq
lr
lr-backbone
weight-decay
num-workers
chunk-size
n-action-steps
dim-model
latent-dim
vision-backbone
device
```

典型训练命令：

```bash
python src/train.py \
  --data-path ./data/nero-dataset \
  --output-dir outputs/nero_act_v3 \
  --batch-size 16 \
  --training-steps 12000 \
  --warmup-steps 1200 \
  --log-freq 100 \
  --save-freq 2000 \
  --lr 5e-5 \
  --lr-backbone 1e-5 \
  --weight-decay 1e-4 \
  --num-workers 8 \
  --chunk-size 32 \
  --n-action-steps 8 \
  --dim-model 512 \
  --latent-dim 64 \
  --vision-backbone resnet18 \
  --device cuda
```

### 11.3 ACT 实验产物

项目中已有多个 ACT 实验目录：

```bash
outputs/nero_act
outputs/nero_act_v2
outputs/nero_act_v3
outputs/nero_act_v4
```

其中 `nero_act_v4` 包含 step 5000 到 step 95000 的多个 checkpoint，说明该策略进行了较完整的长周期训练。

## 12. Pi0.5 训练与推理适配

Pi0.5 推理脚本为：

```bash
src/infer_pi.py
```

Pi0.5 训练命令记录在 `readme.md` 中，训练是在 `/home/qt/Downloads/lerobot` 目录下通过 LeRobot 的训练脚本完成，使用本项目的数据集作为输入。

### 12.1 Pi0.5 训练配置

训练中使用了：

- `policy.type=pi05`
- 本地预训练模型 `/home/qt/Downloads/pi05_base_pretrained`
- 离线 Hugging Face cache
- `bfloat16`
- gradient checkpointing
- freeze vision encoder
- train expert only
- chunk size 32
- n_action_steps 4
- batch size 4
- 40000 training steps

典型训练命令结构如下：

```bash
cd /home/qt/Downloads/lerobot

TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
HF_HOME=/home/qt/Downloads/pushT-so100/.cache/huggingface \
HF_DATASETS_CACHE=/home/qt/Downloads/pushT-so100/.cache/huggingface/datasets \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=/home/qt/Downloads/pushT-so100/data/nero-dataset \
  --policy.type=pi05 \
  --policy.pretrained_path=/home/qt/Downloads/pi05_base_pretrained \
  --output_dir=./outputs/pi05_nero_state15_v1 \
  --job_name=pi05_nero_state15_v1 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.chunk_size=32 \
  --policy.n_action_steps=4 \
  --policy.optimizer_lr=3e-5 \
  --batch_size=4 \
  --steps=40000
```

### 12.2 Pi0.5 推理适配

`infer_pi.py` 做了以下适配：

- 设置 `TRANSFORMERS_OFFLINE=1`
- 加载 LeRobot 训练得到的 PI05 checkpoint
- 从 `model.safetensors` 加载权重
- 使用 `PreTrainedConfig` 恢复配置
- 使用 `make_pre_post_processors` 加载 normalization 和 tokenizer 等处理器
- 在 inference frame 中传入 task 文本
- 使用 Gym 环境闭环执行动作
- 使用 EMA 平滑 TCP 位姿部分
- 夹爪维度不做 EMA，直接使用模型输出

典型推理命令：

```bash
TRANSFORMERS_OFFLINE=1 python src/infer_pi.py \
  --ckpt_path /home/qt/Downloads/lerobot/outputs/pi05_nero_training_v2/checkpoints/005000/pretrained_model \
  --task "grasp the object" \
  --n-action-steps 4 \
  --max-steps 600
```

## 13. 策略推理与视频评估

项目支持三种策略推理：

```bash
src/infer.py       # Diffusion Policy
src/infer_act.py   # ACT Policy
src/infer_pi.py    # Pi0.5 Policy
```

### 13.1 通用推理流程

三类推理脚本的通用流程是：

1. 加载 checkpoint
2. 加载数据集 metadata
3. 根据数据集 stats 构造 preprocessor / postprocessor
4. 创建 MuJoCo Gym 环境
5. 使用 `RecordVideo` 包装环境
6. reset 环境
7. 获取双相机图像和 15 维 state
8. 构造 LeRobot inference frame
9. 调用 policy.select_action
10. postprocess action
11. 执行若干个 action step
12. 对 action 做 EMA 平滑
13. 保存推理视频
14. 输出距离、是否成功、执行步数等信息

### 13.2 Diffusion 推理

典型命令：

```bash
python src/infer.py \
  --ckpt_path outputs/nero_diffusion_official/checkpoints_2026-05-13_15:45/step_15000 \
  --dataset_id data/nero-dataset \
  --env_path "chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/nero_human_env.xml" \
  --video_folder outputs/recorded_videos \
  --n-action-steps 8 \
  --ema-alpha 0.5 \
  --max-steps 500
```

### 13.3 ACT 推理

典型命令：

```bash
python src/infer_act.py \
  --ckpt_path outputs/nero_act_v4/checkpoints_2026-05-13_14:49/step_25000 \
  --dataset_id data/nero-dataset \
  --env_path "chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/nero_human_env.xml" \
  --video_folder outputs/recorded_videos \
  --n-action-steps 8 \
  --ema-alpha 0.7 \
  --max-steps 500
```

### 13.4 视频输出

推理视频保存目录包括：

```bash
outputs/recorded_videos
outputs/recorded_videos_pi05
outputs/recorded_videos_pi05_state15_030000
```

README 中也保存了 GIF 示例：

```bash
assets/show1.gif
assets/show2.gif
```

这些视频和 GIF 用于展示策略推理过程和任务完成效果。

## 14. 实验产物总结

项目已经产生了多组训练输出。

### 14.1 Diffusion 输出

主要目录：

```bash
outputs/nero_diffusion
outputs/nero_diffusion_v2
outputs/nero_diffusion_official
```

包含内容：

- `config.json`
- `model.safetensors`
- `policy_preprocessor.json`
- `policy_postprocessor.json`
- normalizer / unnormalizer safetensors
- TensorBoard events
- step checkpoint
- final_model

### 14.2 ACT 输出

主要目录：

```bash
outputs/nero_act
outputs/nero_act_v2
outputs/nero_act_v3
outputs/nero_act_v4
```

包含内容：

- 多个训练版本
- 多个时间戳 checkpoint 目录
- step checkpoint
- final_model
- TensorBoard events

### 14.3 Pi0.5 输出

Pi0.5 的训练主要位于外部 LeRobot 目录：

```bash
/home/qt/Downloads/lerobot/outputs/
```

本项目中主要保留了 Pi0.5 推理脚本和视频输出目录。

## 15. 文档工作

项目包含中英文 README 和本地操作记录。

### 15.1 README.md

英文 README 说明了：

- 项目背景
- 目录结构
- 环境配置
- 人类演示采集
- 数据处理
- 策略训练
- 评估推理
- Hugging Face 数据集和模型链接
- 推理 GIF 示例
- 常见问题

### 15.2 README-ZH.md

中文 README 提供了对应的中文项目说明，便于中文用户理解项目工作流。

### 15.3 readme.md

本地 `readme.md` 更偏向实际操作记录，记录了当前使用的命令，包括：

- 遥操作采集命令
- Diffusion 训练命令
- ACT 训练命令
- Diffusion 推理命令
- ACT 推理命令
- 数据集 episode 删除命令
- Git 操作备注
- 代理设置
- Pi0.5 训练命令
- Pi0.5 推理命令

## 16. Git 提交历史概览

仓库提交历史显示项目经历了多个阶段。

主要提交节点包括：

```text
Initial commit
data convert
train
train config
eval
video save
readme
example
zh readme
5.14
5.15new
```

其中 5 月 14 日附近的提交是一次重要扩展，包含：

- 引入 NERO 模型资源
- 新增 NERO MuJoCo XML
- 新增数据删除脚本
- 新增 Diffusion 训练脚本
- 新增 ACT 训练和推理脚本
- 更新遥操作环境
- 更新 Gym 环境
- 加入 SO100 旧环境备份

5 月 15 日附近的提交继续围绕 NERO 模型、夹爪和环境稳定性进行调整。

## 17. 当前工作区状态

当前本地工作区存在未提交修改和新增文件。

### 17.1 已修改文件

```text
chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/nero_arm.xml
chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/nero_human_env.xml
delete_episodes.py
readme.md
src/env_gym_ee.py
src/env_human_ee.py
src/infer.py
src/infer_act.py
src/train.py
src/train_diffusion.py
```

### 17.2 新增文件

```text
.cache/
pushT-so100_core.zip
src/augment_observation_state.py
src/fix.py
src/infer_pi.py
src/inspect_mocap_tcp_offset.py
src/state_features.py
src/tcp_frame.py
```

这些未提交内容主要集中在：

- NERO 环境物理参数调试
- TCP action / TCP state 统一
- Pi0.5 推理支持
- 数据集 schema 迁移
- 数据集修复和 episode 删除
- 当前训练与推理命令记录

## 18. 技术亮点

### 18.1 从低维 PushT 到 7DoF 抓取任务的升级

项目不是只停留在早期 2D PushT 控制，而是扩展到 7DoF 机械臂加夹爪的完整抓取任务。这需要同时处理机器人模型、夹爪控制、TCP 表示、物理接触、数据格式和策略输出维度等问题。

### 18.2 统一 TCP action 表示

通过 `tcp_frame.py` 和 `state_features.py`，项目将采集、训练、推理统一到 TCP action 表示上：

```text
[tcp_x, tcp_y, tcp_z, tcp_qw, tcp_qx, tcp_qy, tcp_qz, gripper]
```

这比直接使用 mocap pose 更合理，因为策略学习的是夹爪真实末端的目标位姿。

### 18.3 多策略实验框架

项目同时支持：

- Diffusion Policy
- ACT
- Pi0.5

三类策略共用同一套数据集、观测格式、环境和推理流程，便于横向比较。

### 18.4 数据工程能力完整

项目包含数据采集、数据 schema 迁移、episode 删除、metadata 修复、stats 重算和 cache 清理等工具。这些工作对机器人模仿学习项目非常关键，因为数据质量直接影响策略训练效果。

### 18.5 实验产物完整

项目中保留了大量 checkpoint、TensorBoard 日志和推理视频，说明不仅完成了代码实现，还进行了多轮训练和调参实验。

## 19. 当前存在的问题与风险

### 19.1 README 与当前任务存在命名差异

项目名称和部分 README 仍使用 `PushT`，但当前主线已经扩展到 NERO 抓取任务。建议后续在文档中明确区分：

- 早期 SO100 PushT 阶段
- 当前 NERO grasping 阶段

### 19.2 action 命名存在历史遗留

当前部分 dataset metadata 中 action names 仍显示为 `mocap_x`、`mocap_y` 等历史命名，而代码中已经逐渐迁移到 TCP action。后续应统一检查：

- `meta/info.json`
- `state_features.py`
- `env_human_ee.py`
- `env_gym_ee.py`
- `train_*.py`
- `infer_*.py`

确保所有字段命名和实际语义一致。

### 19.3 当前工作区未提交

当前有较多未提交修改和新增文件。建议在确认可运行后进行一次整理提交，避免实验状态丢失。

### 19.4 训练结果需要进一步量化

目前仓库中有大量 checkpoint 和视频，但缺少统一的评估表格。后续可以增加：

- 每个 checkpoint 的成功率
- 平均完成步数
- 平均末端距离
- 抓取成功次数
- 不同策略对比表

## 20. 后续改进建议

### 20.1 统一项目命名和任务描述

建议将文档中任务名称整理为：

```text
SO100 PushT 阶段：早期实验
NERO Grasp 阶段：当前主线
```

这样可以减少读者对项目名称和当前代码行为之间差异的困惑。

### 20.2 增加自动评估脚本

建议新增 `eval_batch.py`，对多个 checkpoint 自动运行若干 episode，统计：

- success rate
- average reward
- average final distance
- average lifted rate
- average episode length

### 20.3 固化数据 schema

建议明确当前标准数据格式为：

```text
observation.state: 15D = 7D joints + 8D TCP/gripper
action: 8D = TCP/gripper target
images: cam_top + cam_side
fps: 10
```

并将旧数据转换脚本作为兼容工具保留。

### 20.4 清理 outputs 和 checkpoint

当前 `outputs/` 中实验版本较多，建议后续整理：

- 保留最佳 checkpoint
- 删除明显失败或中断实验
- 添加 `EXPERIMENTS.md` 记录每次实验配置和结果

### 20.5 增加环境 sanity check

建议增加一个小脚本，自动检查：

- MuJoCo XML 是否能加载
- camera 是否能渲染
- action step 是否能正常执行
- observation shape 是否符合数据集 metadata
- gripper 是否能开合
- target_block 是否能被检测和抬起

## 21. 总结

`pushT-so100` 项目完成了从 SO100 PushT 仿真任务到 NERO 7DoF 抓取任务的完整扩展。项目不仅搭建了 MuJoCo 环境和机器人模型，还打通了从人类遥操作采集、LeRobot 数据集生成、数据修复、状态/action schema 升级，到 Diffusion、ACT、Pi0.5 多策略训练和推理评估的完整流程。

当前项目的核心成果可以概括为：

1. 建立了可运行的 MuJoCo 机器人操作仿真环境。
2. 实现了手柄遥操作采集 LeRobot 格式数据。
3. 构建了包含双相机图像、关节状态、TCP 状态和夹爪状态的数据集。
4. 将 action 表示升级为 8D TCP 位姿加夹爪控制。
5. 支持 Diffusion Policy、ACT 和 Pi0.5 三类策略训练与推理。
6. 保留了多轮训练 checkpoint、日志和推理视频。
7. 编写了数据集删除、修复、迁移和统计量重算工具。
8. 对 NERO 机械臂、夹爪和接触物理进行了多轮调参。

因此，本项目已经具备机器人模仿学习实验所需的主要工程闭环。后续重点应放在标准化数据 schema、量化评估指标、整理实验记录和稳定最佳 checkpoint 上。
