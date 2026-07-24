# LeRobot NERO runtime

This directory vendors the source tree used by the RoboTwin PI0.5 server. It is based on
Hugging Face LeRobot and contains the local NERO/feature-adapter runtime changes that were
previously kept in a separate unversioned directory.

- Upstream project: <https://github.com/huggingface/lerobot>
- Upstream license: Apache-2.0 (see `LICENSE`)
- Source layout: `src/lerobot`
- Tests: `tests`

The runtime is intentionally kept inside this repository so a fresh clone does not depend on
`/home/.../lerobot_nero_runtime`. Model checkpoints and Hugging Face caches remain external.
