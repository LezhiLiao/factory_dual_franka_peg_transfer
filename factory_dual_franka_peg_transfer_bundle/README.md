# Factory Dual-Franka Peg Transfer Bundle

This bundle contains the task code, desk asset, and RL checkpoint for the
IsaacLab dual-Franka peg-transfer synthetic-data task.

## Contents

- `scripts/synthetic_factory_dual_franka_peg_transfer_thin.py`
  - Recommended RobotWin-style thin task entrypoint.
- `scripts/synthetic_factory_dual_franka_peg_transfer_planned.py`
  - Golden single-file reference version that has the same behavior.
- `internutopia_extension/tasks/synthetic_base_task.py`
  - Checkpoint loading, reset, recording, HDF5, and MP4 helpers.
- `internutopia_extension/tasks/isaac_motion_primitives.py`
  - Reusable motion primitives: hold, joint trajectory, robot2 grasp, pull,
    gripper, and Factory IK.
- `internutopia_extension/tasks/factory_dual_franka_env.py`
  - Dual-Franka Factory scene, desk005 placement, and Gym registration.
- `asset/desk005/`
  - Desk USD and material resource used by the task.
- `checkpoints/Factory/test/nn/Factory.pth`
  - RL-Games Factory peg-insert checkpoint.
- `checkpoints/Factory/test/params/`
  - Agent and env config snapshots from training.

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

## Run

From the repo root:

```bash
source /data/user/miniconda3/etc/profile.d/conda.sh
conda activate internutopia
source /data/user/isaacsim/setup_conda_env.sh
python scripts/synthetic_factory_dual_franka_peg_transfer_thin.py \
  --headless \
  --enable_cameras \
  --checkpoint checkpoints/Factory/test/nn/Factory.pth \
  --video_frame_repeat 1 \
  --hold_steps 8
```

The script writes HDF5 and MP4 files to:

```text
outputs/factory_dual_franka_peg_transfer_thin/
```

## Git Note

`Factory.pth` is about 204 MB. Track it with Git LFS instead of normal Git:

```bash
git lfs track "checkpoints/**/*.pth"
git add .gitattributes checkpoints/Factory/test/nn/Factory.pth
```

The latest verified command produced:

```text
inserted: true
success: true
```
