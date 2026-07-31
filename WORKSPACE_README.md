# NERO PI0.5 + RoboTwin 工作区快速上手

更新时间：2026-07-31

本工作区用于 NERO 双臂任务的数据生成、PI0.5 推理、多视角评测和 View Feature Adapter 训练。主要任务是 `place_two_cubes_box` 与 `beat_block_hammer`。

## 1. 目录

| 路径 | 用途 |
|---|---|
| `/home/qt/Downloads/RoboTwin` | 仿真、任务、评测和 Adapter 脚本，日常入口 |
| `/home/qt/Downloads/lerobot` | PI0.5 训练代码与模型 checkpoint |
| `/home/qt/Downloads/RoboTwin/third_party/lerobot_nero_runtime` | 当前推理使用的本地 PI0.5 runtime |
| `/home/qt/Downloads/RoboTwin/data` | RoboTwin、LeRobot 和多视角数据 |
| `/home/qt/Downloads/RoboTwin/outputs` | Adapter、评测结果与视频 |
| `/home/qt/Downloads/对话记录` | 实验背景、设计和分析文档 |

所有推理都应使用 `qt` 本地代码，不要引用 `/home/chenglong/...`。

## 2. 环境

```bash
cd /home/qt/Downloads/RoboTwin

# RoboTwin 仿真/评测
source /home/qt/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin5090

export ROBOTWIN_LEROBOT_SRC=$PWD/third_party/lerobot_nero_runtime/src
export ROBOTWIN_PI05_SERVER_SRC=$ROBOTWIN_LEROBOT_SRC
export ROBOTWIN_PI05_SERVER_PYTHON=/home/qt/miniforge3/envs/lerobot-nero/bin/python
```

运行 PI0.5 前先用 `nvidia-smi` 确认没有其他训练或推理进程占用大量显存。同一时间只启动一个 PI0.5 server。

## 3. 当前模型

### 单任务 pickplace

基础模型：

```text
/home/qt/Downloads/lerobot/outputs/train/pickplace_joint_sim/checkpoints/010000/pretrained_model
```

已验证有效的单任务 teacher-first Adapter：

```text
/home/qt/Downloads/RoboTwin/outputs/pi05_feature_adapter/train_six_views_teacher_first_20260720/deployment_checkpoint
```

该 Adapter 在 C1-C6 的 100 条/视角评测中将成功率从 `61.5%` 提高到 `72.7%`。

### 双任务 pickplace + hammer

当前推荐模型（不启用 Adapter）：

```text
/home/qt/Downloads/lerobot/outputs/train/three_task_sim/pickplace_hammer_hold3s/checkpoints/024000/pretrained_model
```

不要部署下面的双任务 Adapter：

```text
/home/qt/Downloads/RoboTwin/outputs/pi05_feature_adapter/pickplace_hammer_offline_selection_20260730/step_004500/deployment_checkpoint
```

它在完整闭环评测中把双任务 C1-C6 成功率从 `73.1%` 降到 `16.7%`，仅保留作失败分析。

## 4. 单任务与双任务 Adapter 对比

以下均为 pickplace 闭环成功率。单任务实验每视角 100 条，双任务实验每视角 30 条；因此不要直接比较两组基础模型的绝对高低，但每组内部的 Base/Adapter A/B 结论有效。

| 视角 | 单任务 Base | 单任务 Adapter | 变化 | 双任务 Base 24k | 双任务 Adapter 4.5k | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| C0 canonical | 90% | 92% | +2 pp* | 100% | 100% | 0 pp* |
| C1 yaw -10° | 49% | 61% | +12 pp | 63.3% | 0% | -63.3 pp |
| C2 yaw +10° | 64% | 68% | +4 pp | 56.7% | 3.3% | -53.3 pp |
| C3 pitch -7° | 45% | 57% | +12 pp | 76.7% | 20.0% | -56.7 pp |
| C4 pitch +7° | 71% | 80% | +9 pp | 96.7% | 13.3% | -83.3 pp |
| C5 前移 10 cm | 54% | 83% | +29 pp | 50.0% | 0% | -50.0 pp |
| C6 后移 10 cm | 86% | 87% | +1 pp | 86.7% | 6.7% | -80.0 pp |
| **C1-C6 合计** | **61.5%** | **72.7%** | **+11.2 pp** | **71.7%** | **7.2%** | **-64.4 pp** |

`*` C0 默认绕过 Adapter。单任务 C0 的 2 pp 差异来自当时未固定推理随机数，不能视为 Adapter 收益。

双任务 Adapter 在 Hammer 上同样负优化：C1-C6 从 `74.4%` 降至 `26.1%`。两个任务合计从 `73.1%` 降至 `16.7%`，说明问题来自 Adapter，而不是单个任务的成功判定。

主要原因：

1. 完整残差 `z + delta_z` 修正过强。困难场景筛选中，完整 Adapter 为 `3/20`，将残差缩放到 10% 后恢复到 `18/20`，而 Base 为 `19/20`。
2. 双任务 Base 已有较强的视角鲁棒性，Adapter 改写正确特征的风险大于收益。
3. 当前 Adapter 位于语言 token 拼接之前，只读取图像 token，没有显式任务条件；pickplace 与 Hammer 的梯度会发生冲突。
4. 4.5k checkpoint 只按极小的离线 teacher-velocity/action-chunk 改进选出，没有先通过闭环 rollout 准入测试。
5. 每个任务只有 40 条原始 Adapter 轨迹；Hammer 通过重复 episode 平衡帧数，并未增加场景和失败恢复状态的多样性。
6. Adapter 与冻结的基础 PI0.5 绑定，单任务和双任务 Adapter 对齐的不是同一个主干，不能跨模型混用。

当前结论：单任务 pickplace 使用已验证的 teacher-first Adapter；双任务使用 24k Base 并关闭 Adapter。下一版双任务结构应采用有界任务条件门控：

```text
z_aligned = z + alpha(image, task) * delta_z,  alpha in [0, 0.25]
```

checkpoint 必须先通过固定场景 seed、固定推理 seed 的小规模闭环 A/B，确认不低于 Base 后才能进行完整评测。

## 5. 最短推理流程

先选择一个模型。以下默认使用推荐的双任务基础模型：

```bash
cd /home/qt/Downloads/RoboTwin
export ROBOTWIN_PI05_POLICY_PATH=/home/qt/Downloads/lerobot/outputs/train/three_task_sim/pickplace_hammer_hold3s/checkpoints/024000/pretrained_model
export ROBOTWIN_PI05_SERVER_PYTHON=/home/qt/miniforge3/envs/lerobot-nero/bin/python
export ROBOTWIN_PI05_SERVER_SRC=$PWD/third_party/lerobot_nero_runtime/src
```

终端 1 启动 server：

```bash
policy/lerobot_pi05/run_server.sh
```

终端 2 跑一次 pickplace：

```bash
export ROBOTWIN_PI05_TASK_NAME=place_two_cubes_box
export ROBOTWIN_PI05_TASK_CONFIG=demo_nero_two_cubes
export ROBOTWIN_PI05_SEED=0
export ROBOTWIN_PI05_TEST_NUM=1
policy/lerobot_pi05/run_client.sh
```

切换模型或 Adapter 时，先停止 server，修改 `ROBOTWIN_PI05_POLICY_PATH`，再重新启动。

## 6. C0-C6 多视角评测

不录像、每视角 10 条：

```bash
cd /home/qt/Downloads/RoboTwin
export ROBOTWIN_PI05_POLICY_PATH=/path/to/pretrained_model_or_deployment_checkpoint
export ROBOTWIN_CAMERA_EVAL_EPISODES=10

policy/lerobot_pi05/run_camera_view_eval.sh \
  --policy-inference-seed 123
```

评测单个视角并录像：

```bash
policy/lerobot_pi05/run_camera_view_eval.sh \
  --view C3 \
  --scenario-seed 100000 \
  --policy-inference-seed 123 \
  --record-video
```

Hammer 评测还需使用与训练一致的左右臂指令和采样范围：

```bash
policy/lerobot_pi05/run_camera_view_eval.sh \
  --task-name beat_block_hammer \
  --task-config demo_nero_beat_block_hammer_slow120 \
  --hammer-arm-aware-instruction \
  --hammer-training-support-range \
  --hammer-contact-success \
  --policy-inference-seed 123
```

结果默认保存到 `RoboTwin/outputs/camera_view_eval/<timestamp>/`，其中 `summary.csv` 是汇总，`results.json` 包含 seed 和完整配置。C0 默认绕过 Adapter；只有明确研究 C0 identity 时才使用 `--adapter-on-c0`。

## 7. Adapter 数据与训练

现有配对多视角数据：

```text
Pickplace: /home/qt/Downloads/RoboTwin/data/place_two_cubes_box_adapter_multiview_50
Hammer:    /home/qt/Downloads/RoboTwin/data/beat_block_hammer_adapter_multiview_50
```

单任务训练示例：

```bash
cd /home/qt/Downloads/RoboTwin

python script/train_pi05_view_feature_adapter.py \
  --base-checkpoint /path/to/pretrained_model \
  --cache-index /path/to/train_index.json \
  --shifted-views c1,c2,c3,c4,c5,c6 \
  --adapter-variant legacy \
  --output outputs/pi05_feature_adapter/experiment_name
```

Adapter 只训练第三人称图像 token 的小型残差模块，PI0.5 主干保持冻结。每个 Adapter 与其基础 PI0.5 checkpoint 绑定，不能跨基础模型直接混用。训练后先做离线诊断，再做少量同 seed 闭环 A/B；不要仅凭 loss 选择部署 checkpoint。

## 8. 关键结果与文档

- 单任务 Adapter 分析：`/home/qt/Downloads/对话记录/单任务adapter分析.md`
- Adapter 结构与使用：`/home/qt/Downloads/对话记录/adapter.md`
- 新结构三版迭代：`/home/qt/Downloads/对话记录/adapter新结构三版迭代.md`
- 单任务 100 条 A/B：`RoboTwin/outputs/camera_view_eval/ab_100_20260722/`
- 双任务完整 A/B：`RoboTwin/outputs/pickplace_hammer_adapter_ab_30/20260730_192417/REPORT.md`
- 双任务失败分析：`RoboTwin/outputs/pickplace_hammer_adapter_ab_30/20260730_192417/ANALYSIS_AND_IMPROVEMENT.md`
- 完整 NERO 工作流：`RoboTwin/docs/nero_pi05.md`

## 9. 交接注意事项

- 不修改已有 checkpoint、数据集或正式评测目录；新实验使用新输出目录。
- 正式 A/B 必须固定场景 seed、`--policy-inference-seed`、任务指令和控制参数。
- `deployment_checkpoint` 内可能含指向基础模型的绝对符号链接；移动机器后需要重新构建或修正链接。
- 训练、推理和录像不要并行占用同一张 GPU。
- 长任务使用 `nohup setsid ... > log 2>&1 < /dev/null &`，同时记录 PID、日志和输出目录。
