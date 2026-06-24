# Factory Dual-Franka Peg Transfer Atomic Skills

This is the skill inventory given to the code-generation model.  The generated
script must keep the same structure as `scripts/synthetic_factory_dual_franka_peg_transfer_thin.py`.

## Runtime Contract

- Base class: `DualFrankaPegTransferTask(FactoryPegTransferAtomicSkills, IsaacDualFrankaMotionMixin, SyntheticDataBaseTask)`.
- Scene/env registration:
  - Import `DUAL_TASK_ID`, `configure_factory_dual_franka_env_cfg`, and `register_dual_franka_factory_env`.
  - Call `register_dual_franka_factory_env()` before defining/running the task.
  - In `main(env_cfg, agent_cfg)`, call `configure_factory_dual_franka_env_cfg(env_cfg, args_cli)`.
- Recording:
  - Use `self.record(phase, actions)` inside every skill or low-level motion.
  - Finish with `dataset_path, video_path = self.finish_episode(success)`.
  - HDF5 and MP4 are written under `args_cli.output_dir`.

## Scene Assumptions

- Two Franka arms face each other across `desk005`.
- `franka1` is the checkpoint-controlled insertion robot.
- `franka2` is the scripted extraction robot.
- The fixed asset is the hole; its marker is `"hole"` and maps to `raw.fixed_pos_obs_frame`.
- The held asset is the peg; its marker is `"peg"` and maps to `raw.held_pos`.
- The default checkpoint is `DEFAULT_CHECKPOINT`.

## Skill API

### `insert_peg(obs, robot, hole_marker, max_steps)`

Uses the RL-Games checkpoint policy to insert the held peg into the fixed hole.

Parameters:
- `obs`: observation returned by `self.reset_scene()`.
- `robot`: must be `"franka1"`.
- `hole_marker`: usually `"hole"`.
- `max_steps`: maximum policy steps.

Returns:
- `(current_obs, inserted)` where `inserted` is `bool`.

### `release_lift(robot, lift_height, steps=None, gripper_width=None)`

Opens the selected robot gripper and lifts vertically.

Parameters:
- `robot`: `"franka1"` or `"franka2"`.
- `lift_height`: meters to move upward.
- `steps`: optional IK waypoint count.
- `gripper_width`: optional finger target; default opens based on held asset diameter.

### `return_home(robot, joint_pos=None, gripper_width=None, steps=None)`

Moves the selected robot to its configured home joint position.

Parameters:
- `robot`: `"franka1"` or `"franka2"`.
- `joint_pos`: optional 7-DoF home arm joints.
- `gripper_width`: optional finger target.
- `steps`: optional interpolation steps.

### `grasp(robot, target_marker, pregrasp_distance, move_steps, grasp_z_offset, ...)`

Moves to a pregrasp above a target marker, descends, closes the gripper, and
returns the grasp pose.

Parameters:
- `robot`: currently `"franka2"` for extraction.
- `target_marker`: usually `"peg"`.
- `pregrasp_distance`: vertical approach height in meters.
- `move_steps`: IK waypoint count.
- `grasp_z_offset`: z offset relative to target marker.
- `open_width`: default `0.08`.
- `closed_width`: default `0.0`.

Returns:
- `grasp_pos`, a tensor position used by `extract`.

### `extract(robot, reference_marker, grasp_pos, pull_height, steps, target_quat=None)`

Pulls the grasped peg upward from the hole.

Parameters:
- `robot`: `"franka2"`.
- `reference_marker`: usually `"hole"`.
- `grasp_pos`: output of `grasp`.
- `pull_height`: upward extraction distance in meters.
- `steps`: IK waypoint count.

## Canonical Sequence

```python
obs = self.reset_scene()
self.table_length = self.measure_table_length()
self.prepare_robot2_home_hold(ROBOT2_INITIAL_JOINT_POS)

_, inserted = self.insert_peg(
    obs=obs,
    robot="franka1",
    hole_marker="hole",
    max_steps=self.insert_steps,
)
if inserted:
    self.release_lift(robot="franka1", lift_height=self.release_lift_height)
    self.return_home(robot="franka1")
    grasp_pos = self.grasp(
        robot="franka2",
        target_marker="peg",
        pregrasp_distance=self.approach_height,
        move_steps=self.extract_steps,
        grasp_z_offset=ROBOT2_GRASP_Z_OFFSET,
    )
    self.extract(
        robot="franka2",
        reference_marker="hole",
        grasp_pos=grasp_pos,
        pull_height=self.pull_height,
        steps=self.extract_steps,
    )
```
