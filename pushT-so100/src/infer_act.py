import argparse
import logging
import os
import sys
from pathlib import Path

# Must be set before importing mujoco (via env_gym_ee)
os.environ["MUJOCO_GL"] = "egl"
_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(_REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_REPO_ROOT / ".cache" / "huggingface" / "datasets"))

import numpy as np
import torch
import cv2
from env_gym_ee import PushT
from gymnasium.wrappers import RecordVideo
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
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
    parser = argparse.ArgumentParser(description="Run diffusion policy inference for PushT")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="outputs/nero_diffusion/checkpoints_2026-05-09_18:11/step_4000",
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
        default="outputs/recorded_videos",
        help="Folder to save evaluation videos"
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=4,
        help="Number of actions to execute per inference call (lower = more frequent replanning)"
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.6,
        help="EMA smoothing factor for actions (0=fully smoothed, 1=no smoothing)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Maximum number of steps per episode (default 600 = 60s at 10fps)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.ckpt_path)
    dataset_id = Path(args.dataset_id)
    env_path = args.env_path
    video_folder = args.video_folder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained policy
    policy = ACTPolicy.from_pretrained(ckpt_path.absolute())
    policy.eval()
    policy.to(device)

    # Load dataset metadata and build preprocessors
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
        name_prefix="pusht_eval_video",
    )

    obs, _ = env.reset()

    print("[INFO] Starting inference...", flush=True)
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
                # EMA smoothing to reduce jitter
                if prev_action is not None:
                    #action = args.ema_alpha * action + (1 - args.ema_alpha) * prev_action
                    smoothed = args.ema_alpha * action[:7] + (1 - args.ema_alpha) * prev_action[:7]
                    action = np.concatenate([smoothed, action[7:8]])  # 夹爪直接用模型输出
                prev_action = action.copy()
                obs, reward, terminated, truncated, info = env.step(action)

            if terminated and not truncated:
                print(f"[INFO] Target reached! {info}", flush=True)

    except KeyboardInterrupt:
        print("[INFO] Inference interrupted by user.", flush=True)
    finally:
        print(f"[INFO] Episode done: step={raw_env.current_step}, dist={info.get('dist_to_block','N/A'):.3f}, lifted={info.get('block_lifted','N/A')}", flush=True)
        env.close()
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
