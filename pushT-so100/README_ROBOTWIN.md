# pushT-so100 in the RoboTwin repository

This directory contains the current source tree of the standalone
`pushT-so100` project. It includes the MuJoCo scenes and robot meshes needed by
the simulator, plus data collection, ACT, Diffusion Policy, and PI0.5 scripts.

The project remains operationally independent from RoboTwin. Sharing a Git
repository does not mean sharing a Python environment.

## Environment isolation

Do not install these dependencies into `RoboTwin5090` or `lerobot-nero`.
Create or activate a dedicated environment and run commands from this
directory:

```bash
cd pushT-so100
conda env create -f environment.yml
conda activate lerobot_new
```

The environments used on the source machine were named `pushT` and
`pushT-pi05`. Both used Python 3.10, LeRobot 0.4.4, and MuJoCo. Environment
names are local conventions and can be changed without changing the code. The
shell entry points set `PYTHONNOUSERSITE=1` so packages installed in a user's
global Python directory cannot override the selected Conda environment.

## Workflows

The maintained entry points resolve this directory automatically, so they can
also be called from the parent RoboTwin directory:

```bash
# Record demonstrations.
bash script/record_demonstration_data.sh

# Train a policy after placing a dataset under data/.
bash script/train_policy.sh

# Run inference after placing a checkpoint under outputs/ or overriding the
# paths in the command.
bash script/infer.sh
```

See `readme.md` for the current NERO/PI0.5 command history and
`PROJECT_REPORT.md` for the detailed implementation report.

## Files intentionally excluded

`data/`, `outputs/`, `.cache/`, Python bytecode, and archive files are ignored.
They contained generated datasets, model checkpoints, evaluation videos, and
local caches in the original working directory. Recreate them locally or copy
them through external model/data storage.

The original nested `.git` directory is also intentionally absent. This
directory is now versioned by the parent RoboTwin repository.
