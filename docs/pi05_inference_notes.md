# RoboTwin NERO PI0.5 Inference

当前任务为 `place_two_cubes_box`：右臂将黄色方块放入黑盒，再由左臂将绿色方块放入黑盒。

本项目的 PI0.5 推理采用本机 client/server 架构：RoboTwin client 在 `RoboTwin5090`
环境运行，PI0.5 server 在 LeRobot 环境运行；两者通过本机 gRPC 端口 `127.0.0.1:8081`
通信。

## Current Compatible Checkpoint

当前 RoboTwin NERO 推理适配器需要：16D state、16D action，以及以下三路视觉输入：

```text
observation.images.front
observation.images.left_wrist
observation.images.right_wrist
```

当前可用 checkpoint：

```text
/path/to/16d-nero-pi05-checkpoint/pretrained_model
```

不要使用输入或动作维度不是 16D 的 checkpoint。

## Terminal 1: Start PI0.5 Server

在第一个终端运行，保持该终端不要关闭：

```bash
cd /path/to/RoboTwin

export ROBOTWIN_PI05_SERVER_PYTHON=python
export ROBOTWIN_PI05_SERVER_SRC="$PWD/third_party/lerobot_nero_runtime/src"

./policy/lerobot_pi05/run_server.sh
```

服务监听：

```text
0.0.0.0:8081
```

检查服务是否就绪：

```bash
ss -ltnp | rg ':8081'
```

预期可以看到 Python 进程监听 `*:8081`。

## Terminal 2: Single-Task PI0.5 Inference

在第二个终端运行一次默认推理评测：

```bash
cd /path/to/RoboTwin

export ROBOTWIN_PI05_POLICY_PATH=/path/to/16d-nero-pi05-checkpoint/pretrained_model
export ROBOTWIN_PI05_SERVER_ADDRESS=127.0.0.1:8081
export ROBOTWIN_PI05_TEST_NUM=1

./policy/lerobot_pi05/run_client.sh
```

常用可选参数：

```bash
export ROBOTWIN_PI05_SEED=0
export ROBOTWIN_PI05_FPS=30
export ROBOTWIN_PI05_ASYNC_RTC=false
```

## Terminal 2: Camera-View Robustness Evaluation

服务端保持运行后，在第二个终端评测第三人称相机变化。脚本固定模型、腕部相机和一组共享的
有效场景 seed，只改变 `head_camera`：

| ID | Camera shift |
| --- | --- |
| C1 | yaw -10 deg |
| C2 | yaw +10 deg |
| C3 | pitch -7 deg |
| C4 | pitch +7 deg |
| C5 | move forward 10 cm |
| C6 | move backward 10 cm |

每个视角默认执行 20 次闭环 rollout：

```bash
cd /path/to/RoboTwin

export ROBOTWIN_PI05_POLICY_PATH=/path/to/16d-nero-pi05-checkpoint/pretrained_model
export ROBOTWIN_PI05_SERVER_ADDRESS=127.0.0.1:8081
export ROBOTWIN_CAMERA_EVAL_EPISODES=20
export ROBOTWIN_CAMERA_EVAL_SEED=0

./policy/lerobot_pi05/run_camera_view_eval.sh
```

开始时会打印：

```text
Collecting 20 shared valid seeds from C0...
```

这是预筛选有效场景的阶段，还没有调用模型。出现 `Shared seeds: [...]` 后，脚本依次执行
C1-C6 的策略闭环评测。`No left camera link` 和 `No right camera link` 是当前手动腕部相机的
提示信息，不代表评测失败。

评测使用原始训练数据中的固定 `seen[0]` 指令：

```text
Use the right arm to place the yellow cube into the black box, then use the left arm to place the green cube into the black box.
```

`run_client.sh` 会在任务的两个 `seen` 模板中随机选择一句；动作模型、checkpoint、三路相机、
30 Hz、相对动作处理和直接仿真控制参数与相机评测脚本一致。

PI0.5 权重会在 client 初次连接时加载一次；后续每个 rollout 只重置 episode 状态，不应重复出现
`Loading model from:` 或 `Time taken to put policy on cuda`。

结果逐视角写入以下目录，即使中途停止也会保留已完成视角：

```text
outputs/camera_view_eval/<timestamp>/results.json
outputs/camera_view_eval/<timestamp>/summary.csv
```

## Stop Processes

在评测 client 或 server 所在终端按 `Ctrl-C` 即可停止对应进程。不要关闭 server 后继续运行
client；client 会在连接 `127.0.0.1:8081` 时失败。
