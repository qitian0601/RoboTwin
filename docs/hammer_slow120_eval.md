# Hammer slow120 collection and evaluation snapshot

This snapshot preserves the Hammer setup used for the July 30, 2026 C0
evaluation of the two-task PI0.5 checkpoint at step 24000.

## Collection configuration

Use `demo_nero_beat_block_hammer_slow120`. It records 120 balanced-arm expert
episodes, slows planned motion with a 0.5 time-dilation factor, and holds the
final successful pose for 3 seconds. The Hammer environment also supports
configurable grasp/final holds and a sustained default success check.

The original balanced task samples from these candidate offsets relative to the
table center:

- left X: `[-0.25, -0.06] m`
- right X: `[0.06, 0.25] m`
- Y: `[-0.05, 0.15] m`

Feasible collected slow120 trajectories occupied a narrower empirical support:

- left X: `[-0.25, -0.20] m`
- right X: `[0.18, 0.24] m`
- world Y: `[-0.18, -0.15] m` for table Y `-0.30 m`

## Reproducible evaluation

Start exactly one PI0.5 server with RTC disabled:

```bash
export ROBOTWIN_PI05_POLICY_PATH=/path/to/pretrained_model
export ROBOTWIN_PI05_ASYNC_RTC=false
bash policy/lerobot_pi05/run_server.sh
```

In a second shell, run the pinned evaluation contract:

```bash
export ROBOTWIN_PI05_POLICY_PATH=/path/to/pretrained_model
export ROBOTWIN_PYTHON=/path/to/RoboTwin/python
bash policy/lerobot_pi05/evaluate_hammer_training_range.sh
```

The policy predicts 50 actions, executes 40 before replanning, and gives a new
prediction weight of 1.0 in overlapping chunks. Prompts are selected per episode
from the block side. Success is evaluation-only physical Hammer/block contact,
without alignment, stability, or post-contact hold requirements. Videos are
recorded at every simulator step at 1280x800 and 30 FPS.

The verified step-24000 run completed 10/10 episodes: 5/5 left-arm scenes and
5/5 right-arm scenes. A preceding run over the broader task candidate range was
7/10; all five outer, training-support-like scenes succeeded, while only two of
five near-center scenes succeeded.
