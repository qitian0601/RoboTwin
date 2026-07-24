# PI0.5 View Feature Adapter

## Verified model interface

- Runtime: `third_party/lerobot_nero_runtime/src` (override with `ROBOTWIN_LEROBOT_SRC`)
- Joint checkpoint: `outputs/pi05_exact_code_test/pretrained_model`
- Third-person key: `observation.images.front`
- Wrist keys: `observation.images.left_wrist` and `observation.images.right_wrist`
- Each 224x224 image becomes 256 float32 tokens with width 2048 (16x16 patch grid).
- The Adapter is inserted in `PI05Pytorch.embed_prefix` after vision projection and before image/language concatenation. It is applied only to the third-person key.
- Flow uses `x_t = t * noise + (1 - t) * action`, target `u_t = noise - action`, and prediction `v_t = action_out_proj(suffix_out)`.
- The joint checkpoint uses a 50x16 action chunk. Arm joints 0:14 are relative to the state that produced the chunk; gripper widths 14:16 are absolute. Existing quantile normalization is reused.

## Prepare paired data

To generate the six target shifts from the first 50 episodes of
`place_two_cubes_box_lerobot_v3`, start the resumable background collector:

```bash
cd /path/to/RoboTwin
tools/start_adapter_multiview_collection.sh
```

It writes episodes 0-39 to `train`, episodes 40-49 to `val`, and records C0,
C1-C6, and both wrist cameras at 30 Hz. Progress and the process ID are stored
under `logs/adapter_multiview/`. Completed `_SUCCESS` episodes are skipped on
restart. When collection finishes, the runner automatically exports:

```text
outputs/pi05_feature_adapter/adapter_multiview_cache/train_index.json
outputs/pi05_feature_adapter/adapter_multiview_cache/val_index.json
```

Train on all six shifted views with:

```bash
--cache-index outputs/pi05_feature_adapter/adapter_multiview_cache/train_index.json \
--shifted-views c1,c2,c3,c4,c5,c6
```

The HDF5 export runs in the RoboTwin environment because it already contains `h5py`:

```bash
cd /path/to/RoboTwin
${ROBOTWIN_PYTHON:-python} script/export_pi05_adapter_cache.py \
  --dataset-root data/place_two_cubes_box_multiview_50 \
  --dataset-root data/place_two_cubes_box_multiview_50_pitch10_continuation \
  --split train \
  --output outputs/pi05_feature_adapter/cache
```

The loader interpolates the source 10 Hz commanded actions to the checkpoint's 30 Hz, 50-step horizon.

## Train

```bash
cd /path/to/RoboTwin
export PYTHONPATH="${ROBOTWIN_LEROBOT_SRC:-$PWD/third_party/lerobot_nero_runtime/src}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
${ROBOTWIN_PI05_SERVER_PYTHON:-python} \
  script/train_pi05_view_feature_adapter.py \
  --base-checkpoint outputs/pi05_exact_code_test/pretrained_model \
  --cache-index outputs/pi05_feature_adapter/cache/train_index.json \
  --output outputs/pi05_feature_adapter/train_01 \
  --shifted-views c1,c2 \
  --batch-size 1 \
  --max-steps 3000
```

Only the 2,838,016 Adapter parameters are trainable. Metrics are appended to `train_metrics.jsonl`. The final isolated weights are in `feature_adapter/`; `deployment_checkpoint/` combines them with symlinks to the frozen base model and can be passed directly to the existing inference client.

## Evaluate

Start the qt-local server:

```bash
cd /path/to/RoboTwin
policy/lerobot_pi05/run_server.sh
```

In another terminal, evaluate canonical C0 first, then shifted views:

```bash
cd /path/to/RoboTwin
export ROBOTWIN_PI05_POLICY_PATH=$PWD/outputs/pi05_feature_adapter/train_01/deployment_checkpoint
export ROBOTWIN_CAMERA_EVAL_EPISODES=20
policy/lerobot_pi05/run_camera_view_eval.sh --view C0
policy/lerobot_pi05/run_camera_view_eval.sh
```

Reject an Adapter checkpoint if its C0 success rate is more than 5 percentage points below the frozen base checkpoint. The current paired datasets contain pitch changes only; yaw and camera-translation recovery require additional paired views before those results can be treated as trained coverage.
