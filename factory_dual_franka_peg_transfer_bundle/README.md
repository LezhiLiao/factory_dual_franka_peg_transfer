# Factory Dual-Franka Transfer Bundle

This bundle contains the task code and assets for the IsaacLab dual-Franka
peg-transfer and nut-transfer synthetic-data tasks. The checkpoint is hosted
separately on HuggingFace.

## Contents

- `scripts/synthetic_factory_dual_franka_peg_transfer_thin.py`
  - RobotWin-style thin entrypoint for the dual-Franka peg-transfer task.
- `scripts/synthetic_factory_dual_franka_nut_transfer_thin.py`
  - RobotWin-style thin entrypoint for the dual-Franka nut-transfer task.
  - Franka2 closed gripper width is fixed at `0.01`.
  - Franka2 nut grasp height is shifted down by `5 mm`.
- `scripts/factory_dual_franka_peg_transfer_atomic_skills.py`
  - Shared task-level skills used by both thin entrypoints.
- `internutopia_extension/tasks/synthetic_base_task.py`
  - Checkpoint loading, reset, recording, HDF5, and MP4 helpers.
- `internutopia_extension/tasks/isaac_motion_primitives.py`
  - Reusable motion primitives: hold, joint trajectory, robot2 grasp, pull,
    gripper, wrist twist, release lift, and Factory IK.
- `internutopia_extension/tasks/factory_dual_franka_env.py`
  - Dual-Franka Factory scene, desk005 placement, D455 wrist-camera USD setup,
    and Gym registration for peg and nut tasks.
- `asset/desk005/`
  - Desk USD and material resource used by the task.
- `asset/franka/` and `asset/ur5e_robotiq/`
  - D455 wrist-camera wrapper and payload assets used by `franka1_d455` and
    `franka2_d455` recording modes.

## Expected Repo Layout

Put this bundle's contents at the root of an InternUtopia checkout that also
has:

```text
IsaacLab/
isaacsim installed at /data/user/isaacsim, or ISAAC_PATH set correctly
```

The task script adds these IsaacLab source directories to `sys.path`:

```text
IsaacLab/source/isaaclab
IsaacLab/source/isaaclab_assets
IsaacLab/source/isaaclab_mimic
IsaacLab/source/isaaclab_rl
IsaacLab/source/isaaclab_tasks
```

## Run Peg Transfer

From the repo root:

```bash
source /data/user/miniconda3/etc/profile.d/conda.sh
conda activate internutopia
source /data/user/isaacsim/setup_conda_env.sh
python scripts/synthetic_factory_dual_franka_peg_transfer_thin.py \
  --num_envs 1 \
  --device cuda:0 \
  --headless \
  --disable_fabric \
  --checkpoint checkpoints/Factory/test/nn/Factory.pth \
  --video_frame_repeat 1 \
  --hold_steps 8
```

The script writes HDF5 and MP4 files to:

```text
outputs/factory_dual_franka_peg_transfer_thin/
```

## Run Nut Transfer

From the repo root:

```bash
source /data/user/miniconda3/etc/profile.d/conda.sh
conda activate internutopia
source /data/user/isaacsim/setup_conda_env.sh
python scripts/synthetic_factory_dual_franka_nut_transfer_thin.py \
  --num_envs 1 \
  --device cuda:0 \
  --headless \
  --disable_fabric \
  --checkpoint checkpoints/Factory/test/nn/Factory.pth \
  --record_camera franka2_d455 \
  --video_frame_repeat 1 \
  --output_dir outputs/factory_dual_franka_nut_transfer_rl_debug \
  --wrist_camera_rotate_zyx 0 90 180
```

The script writes HDF5 and MP4 files to the `--output_dir` path.

## Checkpoint

The checkpoint is not stored in this GitHub repository. Download it from:

```text
https://huggingface.co/Heiheiheidashuai/factory_dual_franka_peg_transfer_ckpt
```

Place `Factory.pth` at:

```text
checkpoints/Factory/test/nn/Factory.pth
```

The latest verified command produced:

```text
inserted: true
success: true
```

For nut transfer, the current script records the transfer rollout after Franka1
threads the nut and Franka2 grasps, reverse-twists, lifts, and returns home.
