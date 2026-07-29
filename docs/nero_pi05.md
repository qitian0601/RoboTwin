# NERO 数据生成、PI0.5 推理和 adapter

这份文档记录本仓库中已经验证过的 NERO 工作流。仓库只保存源代码、配置和必要的
机器人模型资产；数据集、模型 checkpoint、adapter cache 和评测输出应在本机生成或
从外部模型存储下载。

## 安装

根据 [`environment/robotwin-nero.yml`](../environment/robotwin-nero.yml) 和
[`environment/lerobot-nero.yml`](../environment/lerobot-nero.yml) 建立两个 Python
环境。需要 NVIDIA GPU、兼容的驱动、ffmpeg 和 Git LFS。RoboTwin 的 Torch 必须在运行
安装脚本之前按目标 GPU 单独安装；通用 requirements 不会再覆盖用户选择的 CUDA wheel。

当前 RTX 5090 工作站验证的是 Python 3.10、Torch 2.10.0+cu128、torchvision
0.25.0+cu128、Warp 1.12.0 和 CuRobo v0.7.8：

```bash
conda env create -f environment/robotwin-nero.yml
conda activate RoboTwin5090
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
bash script/_install.sh

conda deactivate
conda env create -f environment/lerobot-nero.yml
```

其他 GPU 应从 PyTorch 官方安装页选择受该显卡和驱动支持的 wheel，再运行 `_install.sh`。
不要在安装完成后用旧版 requirements 覆盖 Torch。如果主动更换了 Torch，使用
`ROBOTWIN_REINSTALL_PYTORCH3D=1 bash script/_install.sh` 重编译 PyTorch3D。

仓库内置的 `lerobot-nero` 环境固定了已经验证的 PI0.5 依赖，并同时用于模型服务、
Adapter 训练和 RoboTwin 到 LeRobot v3 的离线转换。如果环境名称或安装位置不同，使用
环境变量覆盖默认值：

```bash
export ROBOTWIN_PYTHON=python
export ROBOTWIN_LEROBOT_SRC="$PWD/third_party/lerobot_nero_runtime/src"
export ROBOTWIN_PI05_SERVER_PYTHON=python
export ROBOTWIN_PI05_SERVER_SRC="$PWD/third_party/lerobot_nero_runtime/src"
export ROBOTWIN_LEROBOT_ENV=lerobot-nero
```

克隆后先执行 `git lfs pull && git lfs fsck`。NERO URDF/mesh 由 Git LFS 提供，任务物体
和背景资产则通过以下命令下载；路径生成导致本机 `curobo.yml` 显示为修改是正常现象：

```bash
conda activate RoboTwin5090
bash script/_download_assets.sh
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

实验 runtime 支持以下 Adapter 结构：

- `legacy`：兼容已有 teacher-first checkpoint；
- `pure_teacher_gated_residual`：由全局图像上下文控制的残差 Adapter；
- `multi_scale_dynamic`：动态融合多尺度空间特征；
- `image_routed_moe`：按图像内容路由的多专家 Adapter。

新实验应显式指定结构，并为每种结构使用独立输出目录。例如：

```bash
python script/train_pi05_view_feature_adapter.py \
  --base-checkpoint /path/to/16d_pi05/pretrained_model \
  --cache-index /path/to/merged_adapter_cache/train_index.json \
  --adapter-variant image_routed_moe \
  --adapter-num-experts 4 \
  --output outputs/pi05_feature_adapter/image_routed_moe
```

训练完成后，`deployment_checkpoint/config.json` 会记录 Adapter 结构，权重位于
`deployment_checkpoint/feature_adapter/`。不要把一种结构的权重复制到另一种结构的部署目录。
推理时不需要修改代码，通过选择对应的完整 deployment checkpoint 切换 Adapter：

```bash
export ROBOTWIN_PI05_POLICY_PATH="$PWD/outputs/pi05_feature_adapter/image_routed_moe/deployment_checkpoint"
./policy/lerobot_pi05/run_server.sh
```

切换到另一种 Adapter 时先停止当前 server，修改 `ROBOTWIN_PI05_POLICY_PATH` 后重新启动。
`ROBOTWIN_PI05_FEATURE_ADAPTER=on|off|auto` 只控制当前 checkpoint 中的 Adapter 是否启用，
不会改变 Adapter 结构。`auto` 是默认值；相机评估还可以让 C0 绕过 Adapter，以对照原始模型。

## 快速检查

```bash
python script/test_nero_embodiment.py
python script/validate_nero_task_smoke.py --help
```

建议在新机器上先每个任务生成一个 episode，再进行完整批量生成和模型推理。
