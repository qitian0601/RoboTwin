#!/usr/bin/env python
import pandas as pd
from pathlib import Path
import shutil

DATASET_PATH = Path("/home/qt/Downloads/pushT-so100/data/nero-dataset")
TASKS_PARQUET = DATASET_PATH / "meta/tasks.parquet"

# 查看当前内容
print("=== 当前 tasks.parquet 内容 ===")
df = pd.read_parquet(TASKS_PARQUET)
print(df)
print("index:", df.index.tolist())
print("columns:", df.columns.tolist())
print()

# task 是 index，直接重命名 index
OLD_TASK = "nero_manipulation"
NEW_TASK = "grasp the red cylinder on the table"

df.index = df.index.map(lambda x: NEW_TASK if x == OLD_TASK else x)
df.index.name = "task"  # 保持 index 名称

print("=== 修改后内容 ===")
print(df)

# 备份原文件
shutil.copy(TASKS_PARQUET, str(TASKS_PARQUET) + ".bak")
print(f"\n原文件备份到: {TASKS_PARQUET}.bak")

# 写回
df.to_parquet(TASKS_PARQUET, index=True)
print(f"已保存到: {TASKS_PARQUET}")