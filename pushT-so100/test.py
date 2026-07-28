#!/usr/bin/env python3
"""
delete_episodes.py 集成测试脚本
- 从当前数据集中删除最后 1 个 episode（最小改动）
- 验证 LeRobotDataset 能正常加载并 iterate 几个 batch
- 测试失败时自动从备份恢复
- 测试通过后自动恢复原始数据集（测试完不保留删除结果）

用法：
  python test_delete.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path("./data/nero-dataset")
BACKUP_DIR = Path("./data/nero-dataset_test_backup")


def log(msg, level="INFO"):
    prefix = {"INFO": "ℹ️ ", "OK": "✅", "FAIL": "❌", "WARN": "⚠️ "}.get(level, "  ")
    print(f"{prefix} {msg}")


def check_dataset_integrity(data_dir: Path) -> bool:
    """检查数据集完整性"""
    import pandas as pd
    import numpy as np

    log("检查数据集完整性...")
    info = json.loads((data_dir / "meta/info.json").read_text())
    total_eps = info["total_episodes"]
    total_frames = info["total_frames"]
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    errors = []

    # 1. data parquet 文件数量
    data_files = sorted((data_dir / "data/chunk-000").glob("file-*.parquet"))
    if len(data_files) != total_eps:
        errors.append(f"data parquet 文件数 {len(data_files)} != total_episodes {total_eps}")

    # 2. index 列连续性
    all_idx = []
    for f in data_files:
        df = pd.read_parquet(f, columns=["index", "episode_index"])
        all_idx.extend(df["index"].tolist())
    all_idx_sorted = sorted(all_idx)
    expected = list(range(len(all_idx)))
    if all_idx_sorted != expected:
        errors.append(f"index 列不连续！范围 {min(all_idx)}~{max(all_idx)}，共 {len(all_idx)} 帧，期望 0~{len(all_idx)-1}")
    if len(all_idx) != total_frames:
        errors.append(f"实际帧数 {len(all_idx)} != total_frames {total_frames}")

    # 3. meta/episodes 文件数量及字段
    meta_files = sorted((data_dir / "meta/episodes/chunk-000").glob("file-*.parquet"))
    if len(meta_files) != total_eps:
        errors.append(f"meta/episodes 文件数 {len(meta_files)} != total_episodes {total_eps}")
    for f in meta_files:
        df = pd.read_parquet(f)
        ep = int(df["episode_index"].iloc[0])
        fi = int(df["data/file_index"].iloc[0])
        from_idx = int(df["dataset_from_index"].iloc[0])
        to_idx = int(df["dataset_to_index"].iloc[0])
        meta_fi = int(df["meta/episodes/file_index"].iloc[0])
        expected_fi = int(f.stem.split("-")[1])
        if fi != expected_fi:
            errors.append(f"{f.name}: data/file_index={fi} 但文件编号={expected_fi}")
        if ep != expected_fi:
            errors.append(f"{f.name}: episode_index={ep} 但文件编号={expected_fi}")
        if meta_fi != expected_fi:
            errors.append(f"{f.name}: meta/episodes/file_index={meta_fi} 但文件编号={expected_fi}")
        if to_idx - from_idx <= 0:
            errors.append(f"{f.name}: dataset_from/to_index 异常: {from_idx}~{to_idx}")
        if to_idx > total_frames:
            errors.append(f"{f.name}: dataset_to_index={to_idx} > total_frames={total_frames}")

    # 4. 视频文件数量
    for vk in video_keys:
        vid_dir = data_dir / f"videos/{vk}/chunk-000"
        if vid_dir.exists():
            vids = sorted(vid_dir.glob("file-*.mp4"))
            if len(vids) != total_eps:
                errors.append(f"videos/{vk}: {len(vids)} 个视频 != {total_eps}")

    if errors:
        for e in errors:
            log(e, "FAIL")
        return False
    log(f"完整性检查通过：{total_eps} episodes，{total_frames} 帧", "OK")
    return True


def test_lerobot_load(data_dir: Path) -> bool:
    """用 Python 子进程测试 LeRobotDataset 加载 + 读取几个 batch"""
    log("测试 LeRobotDataset 加载...")
    test_code = f"""
import sys
sys.path.insert(0, '/home/qt/Downloads/lerobot/src')
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

data_dir = '{data_dir.resolve()}'
dataset = LeRobotDataset(
    repo_id='{data_dir.resolve().name}',
    root=data_dir,
    delta_timestamps={{
        'observation.images.cam_top': [-0.1, 0.0],
        'observation.images.cam_side': [-0.1, 0.0],
        'observation.state': [-0.1, 0.0],
        'action': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    }},
)
print(f'Dataset loaded: {{len(dataset)}} frames, {{dataset.num_episodes}} episodes')

loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
count = 0
for batch in loader:
    count += 1
    if count >= 5:
        break
print(f'Successfully iterated {{count}} batches')
"""
    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            log(line, "OK")
        return True
    else:
        log("LeRobotDataset 加载失败！", "FAIL")
        print("--- stdout ---")
        print(result.stdout[-2000:] if result.stdout else "(空)")
        print("--- stderr ---")
        print(result.stderr[-2000:] if result.stderr else "(空)")
        return False


def restore_backup(backup_dir: Path, data_dir: Path):
    """从备份恢复数据集"""
    log(f"正在从备份恢复: {backup_dir} → {data_dir}", "WARN")
    if data_dir.exists():
        shutil.rmtree(data_dir)
    shutil.copytree(backup_dir, data_dir)
    # 清缓存
    import os
    cache = Path.home() / ".cache/huggingface/datasets"
    if cache.exists():
        shutil.rmtree(cache)
    log("恢复完成，缓存已清理", "OK")


def main():
    data_dir = DATA_DIR.resolve()
    if not data_dir.exists():
        log(f"数据集不存在: {data_dir}", "FAIL")
        sys.exit(1)

    print("=" * 60)
    print("LeRobot 数据集删除脚本集成测试")
    print("=" * 60)

    # ── Phase 0: 检查当前数据集健康状态 ──────────────────────────────
    print("\n[Phase 0] 检查当前数据集状态...")
    if not check_dataset_integrity(data_dir):
        log("当前数据集已有问题，请先修复再测试", "FAIL")
        sys.exit(1)
    if not test_lerobot_load(data_dir):
        log("当前数据集 LeRobot 加载已有问题，请先修复", "FAIL")
        sys.exit(1)
    log("当前数据集状态正常", "OK")

    # ── Phase 1: 创建测试备份 ─────────────────────────────────────────
    print("\n[Phase 1] 创建测试备份...")
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    shutil.copytree(data_dir, BACKUP_DIR)
    log(f"备份完成: {BACKUP_DIR}", "OK")

    # ── Phase 2: 执行删除（最后 1 个 episode） ─────────────────────────
    info = json.loads((data_dir / "meta/info.json").read_text())
    last_ep = info["total_episodes"] - 1
    print(f"\n[Phase 2] 测试删除 episode {last_ep}...")

    result = subprocess.run(
        [sys.executable, "delete_episodes.py",
         "--episodes", str(last_ep),
         "--data-dir", str(data_dir),
         "--no-backup"],
        capture_output=True, text=True, timeout=120
    )
    print(result.stdout)
    if result.returncode != 0:
        log("delete_episodes.py 执行失败！", "FAIL")
        print(result.stderr[-1000:])
        restore_backup(BACKUP_DIR, data_dir)
        sys.exit(1)

    # ── Phase 3: 验证完整性 ───────────────────────────────────────────
    print("\n[Phase 3] 验证完整性...")
    # 清缓存（delete_episodes.py 里已清，但以防万一）
    cache = Path.home() / ".cache/huggingface/datasets"
    if cache.exists():
        shutil.rmtree(cache)

    integrity_ok = check_dataset_integrity(data_dir)
    load_ok = test_lerobot_load(data_dir) if integrity_ok else False

    # ── Phase 4: 无论如何都从备份恢复（测试环境不保留删除结果） ──────
    print("\n[Phase 4] 恢复原始数据集...")
    restore_backup(BACKUP_DIR, data_dir)
    shutil.rmtree(BACKUP_DIR)
    log("已恢复原始数据集，测试备份已清理", "OK")

    # ── 结果 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if integrity_ok and load_ok:
        log("所有测试通过！delete_episodes.py 工作正常，可以放心使用。", "OK")
        sys.exit(0)
    else:
        log("测试失败！原始数据集已恢复，请检查上方错误信息。", "FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()