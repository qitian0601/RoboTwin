"""
LeRobot 数据集 episode 删除脚本
用法：
  python delete_episodes.py --episodes 0-9,55-59
  python delete_episodes.py --episodes 3,7,12
  python delete_episodes.py --episodes 0-4 --dry-run
  python delete_episodes.py --episodes 0-4 --data-dir ./data/nero-dataset --no-backup

修复的字段：
  data/chunk-000/file-NNN.parquet      → episode_index, index（全局帧编号从0连续）
  meta/episodes/chunk-000/file-NNN.parquet → episode_index, data/file_index,
                                              videos/*/file_index, meta/episodes/file_index,
                                              dataset_from_index, dataset_to_index
  videos/*/chunk-000/file-NNN.mp4      → 按新编号重命名
  meta/info.json                       → total_episodes, total_frames, splits
  meta/stats.json                      → 重新计算
  ~/.cache/huggingface/datasets/       → 清理 Arrow 缓存
"""
import argparse
import json
import os
import shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


def parse_episodes(ep_str: str) -> list[int]:
    """解析 '0-9,55-59' 或 '3,7,12' 格式"""
    result = set()
    for part in ep_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return sorted(result)


def recompute_stats(data_dir: Path, info: dict) -> dict:
    """从 data parquet 重新计算 meta/stats.json（所有非 video 特征）"""
    chunk_dir = data_dir / "data/chunk-000"
    all_dfs = [pd.read_parquet(f) for f in sorted(chunk_dir.glob("file-*.parquet"))]
    df_all = pd.concat(all_dfs, ignore_index=True)

    stats = {}
    for col, feat in info.get("features", {}).items():
        if feat.get("dtype") == "video" or col not in df_all.columns:
            continue
        arr = np.vstack(df_all[col].tolist()).astype(np.float32)
        stats[col] = {
            "min":  arr.min(axis=0).tolist(),
            "max":  arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std":  arr.std(axis=0).tolist(),
            "count": [float(len(arr))],
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q10": np.quantile(arr, 0.10, axis=0).tolist(),
            "q50": np.quantile(arr, 0.50, axis=0).tolist(),
            "q90": np.quantile(arr, 0.90, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }
    return stats


def rename_with_tmp(files: list[Path], old_to_new: dict[int, int], suffix: str):
    """
    安全重命名：先全部加 _tmp_ 前缀，再按新编号命名，避免同目录下编号冲突。
    files: 已排序的文件列表
    old_to_new: {旧编号: 新编号}
    suffix: '.parquet' 或 '.mp4'
    """
    parent = files[0].parent if files else None
    if not parent:
        return
    # Step 1: 重命名为临时名
    for f in files:
        old_num = int(f.stem.split("-")[1])
        if old_num in old_to_new:
            f.rename(parent / f"_tmp_file-{old_num:03d}{suffix}")
    # Step 2: 从临时名重命名为新编号
    for tmp_f in sorted(parent.glob(f"_tmp_file-*{suffix}")):
        old_num = int(tmp_f.stem.replace("_tmp_file-", ""))
        new_num = old_to_new[old_num]
        tmp_f.rename(parent / f"file-{new_num:03d}{suffix}")


def episode_length_for_delete(data_dir: Path, episode_index: int) -> int:
    """Return episode length even when its data parquet is corrupted."""
    data_file = data_dir / f"data/chunk-000/file-{episode_index:03d}.parquet"
    try:
        return len(pd.read_parquet(data_file))
    except Exception as exc:
        meta_file = data_dir / f"meta/episodes/chunk-000/file-{episode_index:03d}.parquet"
        if not meta_file.exists():
            print(
                f"⚠️  data/file-{episode_index:03d}.parquet 读取失败，且没有对应的 "
                f"meta/episodes 文件；按 0 帧残缺 episode 处理。原错误: {exc}"
            )
            return 0
        meta_df = pd.read_parquet(meta_file)
        if "length" not in meta_df.columns or len(meta_df) == 0:
            raise
        length = int(meta_df["length"].iloc[0])
        print(
            f"⚠️  data/file-{episode_index:03d}.parquet 读取失败，"
            f"使用 meta/episodes 中的 length={length}。原错误: {exc}"
        )
        return length


def delete_episodes(data_dir: Path, episodes_to_delete: set, dry_run: bool = False):
    data_dir = data_dir.resolve()
    delete_set = set(episodes_to_delete)

    info = json.loads((data_dir / "meta/info.json").read_text())
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]

    # ── 读取当前所有 episode 编号 ──────────────────────────────────────
    data_chunk = data_dir / "data/chunk-000"
    data_files = sorted(data_chunk.glob("file-*.parquet"))
    all_ep_nums = [int(f.stem.split("-")[1]) for f in data_files]

    invalid = [e for e in delete_set if e not in all_ep_nums]
    if invalid:
        print(f"❌ 以下 episode 编号不存在: {invalid}")
        return False

    keep_nums = sorted(set(all_ep_nums) - delete_set)
    old_to_new = {old: new for new, old in enumerate(keep_nums)}

    total_before = info["total_frames"]
    del_frames = sum(episode_length_for_delete(data_dir, e) for e in delete_set)

    print(f"数据集: {data_dir}")
    print(f"当前: {len(all_ep_nums)} episodes, {total_before} 帧")
    print(f"删除: {sorted(delete_set)} ({len(delete_set)} episodes, {del_frames} 帧)")
    print(f"保留: {len(keep_nums)} episodes, {total_before - del_frames} 帧（预估）")

    if dry_run:
        print("\n[DRY RUN] 不会修改任何文件。")
        return True

    # ── Step 1: 删除不需要的文件 ─────────────────────────────────────
    print("\n[1/6] 删除文件...")
    for ep in sorted(delete_set):
        f = data_chunk / f"file-{ep:03d}.parquet"
        if f.exists(): f.unlink(); print(f"  ✗ data/file-{ep:03d}.parquet")
        for vk in video_keys:
            vf = data_dir / f"videos/{vk}/chunk-000/file-{ep:03d}.mp4"
            if vf.exists(): vf.unlink(); print(f"  ✗ videos/{vk}/file-{ep:03d}.mp4")
        mf = data_dir / f"meta/episodes/chunk-000/file-{ep:03d}.parquet"
        if mf.exists(): mf.unlink(); print(f"  ✗ meta/episodes/file-{ep:03d}.parquet")

    # ── Step 2: 修复 data parquet 内部字段（episode_index, index） ────
    print("\n[2/6] 修复 data parquet 字段...")
    global_idx = 0
    for old_ep in keep_nums:
        f = data_chunk / f"file-{old_ep:03d}.parquet"
        df = pd.read_parquet(f)
        n = len(df)
        df["episode_index"] = old_to_new[old_ep]
        df["index"] = list(range(global_idx, global_idx + n))
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), f)
        global_idx += n
    total_frames = global_idx
    print(f"  总帧数: {total_frames}")

    # ── Step 3: 重命名 data parquet ───────────────────────────────────
    print("\n[3/6] 重命名 data parquet...")
    rename_with_tmp(sorted(data_chunk.glob("file-*.parquet")), old_to_new, ".parquet")

    # ── Step 4: 修复 meta/episodes parquet 字段 + 重命名 ──────────────
    print("\n[4/6] 修复 meta/episodes 字段并重命名...")
    # 从修复后的 data parquet 读取各 episode 的帧范围（用新编号找文件）
    ep_frame_bounds = {}
    for old_ep in keep_nums:
        new_ep = old_to_new[old_ep]
        df = pd.read_parquet(data_chunk / f"file-{new_ep:03d}.parquet", columns=["index"])
        ep_frame_bounds[old_ep] = (int(df["index"].min()), int(df["index"].max()) + 1)

    meta_ep_chunk = data_dir / "meta/episodes/chunk-000"
    for old_ep in keep_nums:
        new_ep = old_to_new[old_ep]
        f = meta_ep_chunk / f"file-{old_ep:03d}.parquet"
        df = pd.read_parquet(f)
        from_idx, to_idx = ep_frame_bounds[old_ep]
        df["episode_index"] = new_ep
        df["data/file_index"] = new_ep
        df["meta/episodes/file_index"] = new_ep
        df["dataset_from_index"] = from_idx
        df["dataset_to_index"] = to_idx
        for vk in video_keys:
            col = f"videos/{vk}/file_index"
            if col in df.columns:
                df[col] = new_ep
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), f)
        print(f"  episode {old_ep}→{new_ep}: from={from_idx}, to={to_idx}")
    rename_with_tmp(sorted(meta_ep_chunk.glob("file-*.parquet")), old_to_new, ".parquet")

    # ── Step 5: 重命名视频文件 ────────────────────────────────────────
    print("\n[5/6] 重命名视频文件...")
    for vk in video_keys:
        vchunk = data_dir / f"videos/{vk}/chunk-000"
        if vchunk.exists():
            rename_with_tmp(sorted(vchunk.glob("file-*.mp4")), old_to_new, ".mp4")
            print(f"  ✓ videos/{vk}")

    # ── Step 6: 更新 meta/info.json 和 meta/stats.json ────────────────
    print("\n[6/6] 更新 meta/info.json 和 stats.json...")
    info["total_episodes"] = len(keep_nums)
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{len(keep_nums)}"}
    (data_dir / "meta/info.json").write_text(json.dumps(info, indent=2))

    stats = recompute_stats(data_dir, info)
    (data_dir / "meta/stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  total_episodes={len(keep_nums)}, total_frames={total_frames}")

    # ── 清理 HuggingFace Arrow 缓存 ───────────────────────────────────
    for cache_path in [
        Path.home() / ".cache/huggingface/datasets",
        Path.home() / f".cache/huggingface/lerobot/{data_dir.name}",
        data_dir.parent.parent / ".cache/huggingface/datasets",
        data_dir.parent.parent / f".cache/huggingface/lerobot/{data_dir.name}",
    ]:
        if cache_path.exists():
            try:
                shutil.rmtree(cache_path)
                print(f"  🗑  清理缓存: {cache_path}")
            except OSError as exc:
                if exc.errno == 30 or "Read-only file system" in str(exc):
                    print(f"  ⚠️  缓存目录只读，跳过清理: {cache_path}")
                else:
                    raise

    print(f"\n🎉 完成！保留 {len(keep_nums)} 个 episodes，共 {total_frames} 帧")
    return True


def main():
    parser = argparse.ArgumentParser(description="删除 LeRobot 数据集中指定 episodes")
    parser.add_argument("--data-dir", default="./data/nero-dataset", help="数据集路径")
    parser.add_argument("--episodes", required=True, help="要删除的 episode，如 '0-9,55-59' 或 '3,7,12'")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不修改文件")
    parser.add_argument("--no-backup", action="store_true", help="跳过备份（不推荐）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ 数据集路径不存在: {data_dir}")
        return

    episodes_to_delete = set(parse_episodes(args.episodes))
    print("=" * 60)

    if not args.dry_run and not args.no_backup:
        backup_dir = data_dir.parent / f"{data_dir.name}_backup"
        if backup_dir.exists():
            print(f"⚠️  备份目录已存在，跳过备份: {backup_dir}")
        else:
            print(f"📦 备份到 {backup_dir} ...")
            shutil.copytree(data_dir, backup_dir)
            print(f"✅ 备份完成")
    print("=" * 60)

    delete_episodes(data_dir, episodes_to_delete, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
