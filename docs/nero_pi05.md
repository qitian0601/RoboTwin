# NERO 数据生成、PI0.5 推理和 adapter

这份文档记录本仓库中已经验证过的 NERO 工作流。仓库只保存源代码、配置和必要的
机器人模型资产；数据集、模型 checkpoint、adapter cache 和评测输出应在本机生成或
从外部模型存储下载。

## 安装

先按照 [`script/_install.sh`](../script/_install.sh) 安装 RoboTwin 依赖，再根据
[`environment/robotwin-nero.yml`](../environment/robotwin-nero.yml) 和
[`environment/lerobot-nero.yml`](../environment/lerobot-nero.yml) 建立两个 Python
环境。需要 NVIDIA GPU、CUDA、ffmpeg 和 CuRobo v0.7.8；不同机器的 CUDA/PyTorch
组合应保持兼容。

```bash
conda env create -f environment/robotwin-nero.yml
conda env create -f environment/lerobot-nero.yml
conda run -n RoboTwin5090 python -m pip install -r script/requirements.txt
bash script/_install.sh
```

如果环境名称或安装位置不同，用环境变量覆盖默认值：

```bash
export ROBOTWIN_PYTHON=python
export ROBOTWIN_LEROBOT_SRC="$PWD/third_party/lerobot_nero_runtime/src"
export ROBOTWIN_PI05_SERVER_PYTHON=python
export ROBOTWIN_PI05_SERVER_SRC="$PWD/third_party/lerobot_nero_runtime/src"
```

RoboTwin 的 NERO 资产位于 `assets/embodiments/nero`。其中 URDF 的关节链保持一致，
`nero_no_arm_collision.urdf` 用于 SAPIEN 渲染，CuRobo 使用相应的规划配置。若仓库
被放到其他路径，使用模板重新生成 CuRobo 配置：

```bash
python script/update_embodiment_config_path.py
```

## 生成三个 NERO 任务的数据

当前批处理入口会生成：

- `beat_block_hammer` / `demo_nero_beat_block_hammer_c0`
- `click_bell` / `demo_nero_click_bell_c0`
- `pick_dual_bottles` / `demo_nero_pick_dual_bottles_c0`

```bash
./run_new_tasks_adapter_data.sh all 50 40
```

参数依次为任务（`all` 或单个任务名）、每个任务的总 episode 数和训练 episode 数。
脚本会生成 RoboTwin HDF5、LeRobot v3 数据以及 C0-C6 多视角 adapter cache；这些
结果默认写入 `data/`、`outputs/` 和 `logs/`，均不会提交到 Git。

只生成一个任务时，例如：

```bash
./run_new_tasks_adapter_data.sh click_bell 10 8
```

## PI0.5 推理

PI0.5 client 在 RoboTwin 环境中运行，server 使用本仓库随附的 LeRobot NERO runtime。
模型需要由用户自行下载或上传，并且必须与 adapter 的 16 维 state/action 接口匹配。

终端一：

```bash
export ROBOTWIN_PI05_POLICY_PATH=/path/to/pretrained_model
export ROBOTWIN_PI05_SERVER_PYTHON=python
export ROBOTWIN_PI05_SERVER_SRC="$PWD/third_party/lerobot_nero_runtime/src"
./policy/lerobot_pi05/run_server.sh
```

终端二：

```bash
export ROBOTWIN_PI05_POLICY_PATH=/path/to/pretrained_model
export ROBOTWIN_PI05_SERVER_ADDRESS=127.0.0.1:8081
./policy/lerobot_pi05/run_client.sh
```

已有 adapter 需要放在 deployment checkpoint 的 `feature_adapter/` 目录，或通过
`--adapter-checkpoint` 传给训练脚本。不要直接上传当前机器生成的
`deployment_checkpoint` 符号链接；在新机器上用新的 base checkpoint 重新生成部署目录。

## 训练 adapter

先运行上面的三任务数据脚本，再用生成的 `train_index.json` 或 `val_index.json` 调用：

```bash
python script/train_pi05_view_feature_adapter.py \
  --base-checkpoint /path/to/16d_pi05/pretrained_model \
  --cache-index data/click_bell_adapter_multiview_50/train_index.json \
  --output outputs/pi05_feature_adapter/click_bell
```

多任务训练时，应把三任务 cache 合并后的 index 作为 `--cache-index`，不要误用单任务
示例路径。训练会保存 adapter 权重和训练配置，但不会保存完整 PI0.5 base model。

## 快速检查

```bash
python script/test_nero_embodiment.py
python script/validate_nero_task_smoke.py --help
```

建议在新机器上先每个任务生成一个 episode，再进行完整批量生成和模型推理。
