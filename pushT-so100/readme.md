遥操作：cd /home/qt/Downloads/pushT-so100

python src/env_human_ee.py \
  --repo_id ./data/nero-dataset \
  --fps 10 \
  --move_speed 0.05 \
  --rot_speed 0.3  按B开始录制，结束后按B保存

训练：cd /home/qt/Downloads/pushT-so100

python src/train_diffusion.py --data-path ./data/nero-dataset --output-dir outputs/nero_diffusion_v2 --batch-size 16 --training-steps 15000 --warmup-steps 1000 --log-freq 50 --save-freq 2000 --lr 1e-4 --weight-decay 1e-5 --num-workers 8 --n-obs-steps 2 --horizon 16 --n-action-steps 8 --vision-backbone resnet18 --device cuda

python src/train.py --data-path ./data/nero-dataset --output-dir outputs/nero_act_v3 --batch-size 16 --training-steps 12000 --warmup-steps 1200 --log-freq 100 --save-freq 2000 --lr 5e-5 --lr-backbone 1e-5 --weight-decay 1e-4 --num-workers 8 --chunk-size 32 --n-action-steps 8 --dim-model 512 --latent-dim 64 --vision-backbone resnet18 --device cuda


推理：cd /home/qt/Downloads/pushT-so100

python src/infer.py \
  --ckpt_path outputs/nero_diffusion_official/checkpoints_2026-05-13_15:45/step_15000 \
  --dataset_id data/nero-dataset \
  --env_path "chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/nero_human_env.xml" \
  --video_folder outputs/recorded_videos \
  --n-action-steps 8 \
  --ema-alpha 0.5 \
  --max-steps 500
python src/infer_act.py   --ckpt_path outputs/nero_act_v4/checkpoints_2026-05-13_14:49/step_25000   --dataset_id data/nero-dataset   --env_path "chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/nero_human_env.xml"   --video_folder outputs/recorded_videos   --n-action-steps 8   --ema-alpha 0.7   --max-steps 500

  
  ckpt_path需要修改


增删改查数据集：# 预览要删除的内容（不修改任何文件）
python delete_episodes.py --episodes 0-9,55-59 --dry-run

# 正式删除（自动备份 + 修复所有字段 + 清理缓存）
python delete_episodes.py --episodes 44

# 指定数据集路径
python delete_episodes.py --episodes 5,10,15 --data-dir ./data/nero-dataset

# 跳过备份（已有备份时使用）
python delete_episodes.py --episodes 0-4 --no-backup



git add .
git commit -m " "
git reflog 完整操作记录

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

pi训练：
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
    --policy.scheduler_warmup_steps=1000 \
    --policy.scheduler_decay_steps=40000 \
    --policy.scheduler_decay_lr=3e-6 \
    --wandb.enable=false \
    --batch_size=4 \
    --num_workers=8 \
    --steps=40000 \
    --save_freq=5000 \
    --log_freq=100

推理：   
cd /home/qt/Downloads/pushT-so100

TRANSFORMERS_OFFLINE=1 python src/infer_pi.py \
    --ckpt_path /home/qt/Downloads/lerobot/outputs/pi05_nero_training_v2/checkpoints/005000/pretrained_model \
    --task "grasp the object" \
    --n-action-steps 4 \
    --max-steps 600