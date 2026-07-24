python script/collect_data.py beat_block_hammer demo_nero_smoke   直接生成轨迹
python -u script/debug_nero_full_task.py --seeds 1 --render-freq 1  在仿真中实时规划生成轨迹
cd RoboTwin
conda activate RoboTwin5090
conda run --no-capture-output -n RoboTwin5090 python -u script/collect_data.py place_two_cubes_box demo_nero_two_cubes

conda run --no-capture-output -n lerobot python tools/convert_robotwin_to_lerobot_v3.py \
  --input-dir data/place_two_cubes_box/demo_nero_two_cubes \
  --output-dir data/place_two_cubes_box_lerobot_v3 \
  --repo-id place_two_cubes_box_lerobot_v3  转换格式脚本

./run_collect_200.sh  批量采集脚本
