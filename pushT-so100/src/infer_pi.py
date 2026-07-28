import argparse
import logging
import os
import sys
from pathlib import Path

# Must be set before importing mujoco (via env_gym_ee)
os.environ["MUJOCO_GL"] = "egl"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # ← Pi0.5 需要
_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(_REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_REPO_ROOT / ".cache" / "huggingface" / "datasets"))

import numpy as np
import torch
from env_gym_ee import PushT
from gymnasium.wrappers import RecordVideo
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # ← 改这里
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame
from state_features import OBS_STATE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Pi0.5 policy inference for PushT")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="outputs/pi05_nero_training/checkpoints_*/final_model",  # ← 改默认路径
        help="Path to the pretrained checkpoint directory"
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="data/nero-dataset",
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--env_path",
        type=str,
        default="chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/nero_human_env.xml",
        help="Path to the MuJoCo XML environment file"
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default="outputs/recorded_videos_pi05",  # ← 改文件夹名
        help="Folder to save evaluation videos"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="grasp the object",  # ← Pi0.5 新增参数
        help="Task description for Pi0.5 policy"
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=4,
        help="Number of actions to execute per inference call"
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.6,
        help="EMA smoothing factor for actions"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Maximum number of steps per episode"
    )

    return parser.parse_args()


def load_pi05_checkpoint(ckpt_path: Path, device: torch.device) -> PI05Policy:
    """Load a LeRobot-trained PI05 checkpoint without OpenPI key remapping."""
    ckpt_path = ckpt_path.absolute()
    model_file = ckpt_path / "model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"Missing model.safetensors in checkpoint: {ckpt_path}")

    config = PreTrainedConfig.from_pretrained(ckpt_path)
    config.device = str(device)
    policy = PI05Policy(config)
    PreTrainedPolicy._load_as_safetensor(
        policy,
        str(model_file),
        map_location=str(device),
        strict=False,
    )
    policy.eval()
    policy.to(device)
    print(f"[INFO] Loaded LeRobot PI05 checkpoint: {ckpt_path}", flush=True)
    return policy


def main():
    args = parse_args()

    ckpt_path = Path(args.ckpt_path)
    dataset_id = Path(args.dataset_id)
    env_path = args.env_path
    video_folder = args.video_folder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load LeRobot-trained policy checkpoint.
    policy = load_pi05_checkpoint(ckpt_path, device)

    # Load dataset metadata and checkpoint preprocessors.
    # The checkpoint preprocessor stores the local tokenizer path and normalization stats.
    dataset_metadata = LeRobotDatasetMetadata(dataset_id.absolute())
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        dataset_stats=dataset_metadata.stats,
        pretrained_path=ckpt_path,
    )

    # Create environment with video recording
    raw_env = PushT(xml_path=env_path, render_mode="rgb_array", max_steps=args.max_steps)
    env = RecordVideo(
        raw_env,
        video_folder=video_folder,
        episode_trigger=lambda x: True,
        name_prefix="pusht_pi05_eval",  # ← 改前缀
    )

    obs, _ = env.reset()

    print("[INFO] Starting Pi0.5 inference...", flush=True)
    print(f"[INFO] Task: {args.task}", flush=True)
    print(f"[INFO] n_action_steps={args.n_action_steps}, ema_alpha={args.ema_alpha}", flush=True)
    terminated = False
    truncated = False
    prev_action = None

    try:
        while not terminated and not truncated:
            step = raw_env.current_step
            if step % 50 == 0:
                print(f"[INFO] Step {step}/{raw_env.max_steps}", flush=True)

            # Run policy inference
            with torch.no_grad():
                obs_flat = {
                    "cam_top": obs["cam_top"],
                    "cam_side": obs["cam_side"],
                }
                for i, name in enumerate(OBS_STATE_NAMES):
                    obs_flat[name] = obs["observation.state"][i]

                obs_frame = build_inference_frame(
                    observation=obs_flat,
                    ds_features=dataset_metadata.features,
                    device=device,
                    task=args.task,  # ← Pi0.5 必须传 task
                )
                obs_tensor = preprocess(obs_frame)
                actions_sequence = policy.select_action(obs_tensor)
                actions_sequence = postprocess(actions_sequence)

            # Execute n_action_steps actions per inference call with EMA smoothing
            n_exec = min(args.n_action_steps, len(actions_sequence))
            for i in range(n_exec):
                if terminated or truncated:
                    break
                action = actions_sequence[i].cpu().numpy()
                # EMA smoothing
                if prev_action is not None:
                    smoothed = args.ema_alpha * action[:7] + (1 - args.ema_alpha) * prev_action[:7]
                    action = np.concatenate([smoothed, action[7:8]])
                prev_action = action.copy()
                obs, reward, terminated, truncated, info = env.step(action)

            if terminated and not truncated:
                print(f"[INFO] Target reached! {info}", flush=True)

    except KeyboardInterrupt:
        print("[INFO] Inference interrupted by user.", flush=True)
    finally:
        print(f"[INFO] Episode done: step={raw_env.current_step}, dist={info.get('dist_to_block','N/A'):.3f}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
