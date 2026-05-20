"""Reusable IsaacLab motion primitives for synthetic manipulation tasks."""

from __future__ import annotations

import torch

import isaacsim.core.utils.torch as torch_utils
from isaaclab_tasks.direct.factory import factory_control


class IsaacDualFrankaMotionMixin:
    """RobotWin-style primitive layer for the current dual-Franka Factory env.

    The mixin expects ``self.raw`` to be a FactoryEnv-like object with
    ``_robot`` as Franka1 and ``_robot2`` as Franka2.  Task classes keep the
    high-level sequence; this mixin owns low-level target holds, IK stepping,
    gripper interpolation, and deterministic peg carrying after grasp.
    """

    robot2_ik_substeps: int
    hold_steps: int
    gripper_steps: int

    def prepare_robot2_home_hold(self, joint_pos, gripper_width: float = 0.04):
        env_ids = self.env_ids()
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        self._robot2_hold_joint_pos = self.raw._robot2.data.joint_pos[env_ids].clone()
        self._robot2_hold_joint_pos[:, 0:7] = self.tensor_joint_pos(joint_pos)
        self._robot2_hold_joint_pos[:, 7:] = gripper_width
        self.raw.robot2_ctrl_target_joint_pos[env_ids, :] = self._robot2_hold_joint_pos
        for _ in range(max(1, int(self.hold_steps))):
            self.apply_robot2_hold()
            self.raw.step_sim_no_action()

    def run_checkpoint_policy_with_holds(self, obs, phase: int, max_steps: int, stop_on_success: bool = True):
        current_obs = obs
        for _ in range(max(1, int(max_steps))):
            self.apply_hold_targets()
            current_obs, _, _, _, actions = self.rl_skill.step(current_obs)
            self.apply_hold_targets()
            self.record(phase, actions)
            if stop_on_success and self.check_success(require=False):
                break
        return current_obs

    def open_gripper_and_move_joints(self, robot: str, joint_pos, gripper_width: float, steps: int, phase: int):
        articulation, ctrl_target = self.robot_handles(robot)
        env_ids = self.env_ids()
        current = articulation.data.joint_pos[env_ids].clone()
        open_target = current.clone()
        open_target[:, 7:] = gripper_width
        self.execute_joint_trajectory(articulation, open_target, ctrl_target, env_ids, self.gripper_steps, phase)

        target = open_target.clone()
        target[:, 0:7] = self.tensor_joint_pos(joint_pos)
        target[:, 7:] = gripper_width
        self.execute_joint_trajectory(articulation, target, ctrl_target, env_ids, steps, phase)
        self.hold_targets(self.hold_steps, phase)

    def robot2_grasp_actor(
        self,
        peg_pos,
        target_quat,
        pregrasp_height: float,
        grasp_z_offset: float,
        move_steps: int,
        pregrasp_phase: int,
        descend_phase: int,
        grasp_phase: int,
        open_width: float = 0.08,
        closed_width: float = 0.0,
    ):
        pregrasp_pos = peg_pos.clone()
        pregrasp_pos[:, 2] += pregrasp_height
        grasp_pos = peg_pos.clone()
        grasp_pos[:, 2] += grasp_z_offset

        self.robot2_move_to_pose(pregrasp_pos, target_quat, move_steps, pregrasp_phase, gripper_width=open_width)
        self.hold_targets(self.hold_steps, pregrasp_phase)

        descend_pos = pregrasp_pos.clone()
        descend_pos[:, 2] = grasp_pos[:, 2]
        self.robot2_move_to_pose(descend_pos, target_quat, move_steps, descend_phase, gripper_width=open_width)
        self.hold_targets(self.hold_steps, descend_phase)

        self.robot2_set_gripper(closed_width, self.gripper_steps, grasp_phase)
        self.capture_peg_grasp_offset()
        self.hold_targets(self.hold_steps, grasp_phase)
        return grasp_pos

    def robot2_pull_actor(self, hole_pos, grasp_pos, target_quat, pull_height: float, steps: int, phase: int):
        pull_pos = grasp_pos.clone()
        pull_pos[:, 0:2] = hole_pos[:, 0:2]
        pull_pos[:, 2] = grasp_pos[:, 2] + pull_height
        self.robot2_move_to_pose(pull_pos, target_quat, steps, phase, gripper_width=0.0, carry_peg=True)
        self.hold_targets(self.hold_steps, phase, carry_peg=True)

    def robot2_move_to_pose(
        self, target_pos, target_quat, steps: int, phase: int, gripper_width: float, carry_peg: bool = False
    ):
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        start = self.raw.robot2_fingertip_midpoint_pos.clone()
        total_steps = max(1, int(steps) * max(1, int(self.robot2_ik_substeps)))
        env_ids = self.env_ids()
        for i in range(1, total_steps + 1):
            alpha = i / total_steps
            waypoint = torch.lerp(start, target_pos, self.smoothstep(alpha))
            joint_target = self.robot2_solve_ik_joint_target(waypoint, target_quat)
            joint_target[:, 7:] = gripper_width
            self.set_robot2_joint_target(joint_target, env_ids)
            if carry_peg:
                self.carry_peg_with_robot2()
            self.apply_hold_targets(skip_articulation=self.raw._robot2)
            self.raw.step_sim_no_action()
            self.record(phase, self.zero_action())
        self._robot2_hold_joint_pos = self.raw._robot2.data.joint_pos[env_ids].clone()

    def robot2_set_gripper(self, width: float, steps: int, phase: int):
        env_ids = self.env_ids()
        target = self.raw._robot2.data.joint_pos[env_ids].clone()
        target[:, 7:] = width
        self.execute_joint_trajectory(
            self.raw._robot2,
            target,
            self.raw.robot2_ctrl_target_joint_pos,
            env_ids,
            steps,
            phase,
        )

    def execute_joint_trajectory(
        self,
        articulation,
        target_joint_pos,
        ctrl_target_joint_pos,
        env_ids,
        steps: int,
        phase: int,
        carry_peg: bool = False,
    ):
        start_joint_pos = articulation.data.joint_pos[env_ids].clone()
        target_joint_pos = target_joint_pos.clone()
        joint_vel = torch.zeros_like(target_joint_pos)
        n_steps = max(1, int(steps))
        for i in range(1, n_steps + 1):
            joint_pos = torch.lerp(start_joint_pos, target_joint_pos, self.smoothstep(i / n_steps))
            ctrl_target_joint_pos[env_ids, :] = joint_pos
            with torch.inference_mode():
                articulation.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
                articulation.set_joint_position_target(ctrl_target_joint_pos[env_ids], env_ids=env_ids)
            if carry_peg:
                self.carry_peg_with_robot2()
            self.apply_hold_targets(skip_articulation=articulation)
            self.raw.step_sim_no_action()
            self.record(phase, self.zero_action())
        articulation.reset()
        if articulation is self.raw._robot:
            self._franka1_hold_joint_pos = target_joint_pos.clone()
        elif articulation is self.raw._robot2:
            self._robot2_hold_joint_pos = target_joint_pos.clone()

    def hold_targets(self, steps: int, phase: int, carry_peg: bool = False):
        for _ in range(max(0, int(steps))):
            if carry_peg:
                self.carry_peg_with_robot2()
            self.apply_hold_targets()
            self.raw.step_sim_no_action()
            self.record(phase, self.zero_action())

    def apply_hold_targets(self, skip_articulation=None):
        env_ids = self.env_ids()
        if self._franka1_hold_joint_pos is not None and skip_articulation is not self.raw._robot:
            self.apply_joint_hold(
                self.raw._robot,
                self._franka1_hold_joint_pos,
                self.raw.ctrl_target_joint_pos,
                env_ids,
            )
        if self._robot2_hold_joint_pos is not None and skip_articulation is not self.raw._robot2:
            self.apply_joint_hold(
                self.raw._robot2,
                self._robot2_hold_joint_pos,
                self.raw.robot2_ctrl_target_joint_pos,
                env_ids,
            )

    def apply_robot2_hold(self):
        if self._robot2_hold_joint_pos is None:
            return
        self.apply_joint_hold(
            self.raw._robot2,
            self._robot2_hold_joint_pos,
            self.raw.robot2_ctrl_target_joint_pos,
            self.env_ids(),
        )

    def apply_joint_hold(self, articulation, target_joint_pos, ctrl_target_joint_pos, env_ids):
        joint_vel = torch.zeros_like(target_joint_pos)
        ctrl_target_joint_pos[env_ids, :] = target_joint_pos
        with torch.inference_mode():
            articulation.write_joint_state_to_sim(target_joint_pos, joint_vel, env_ids=env_ids)
            articulation.set_joint_position_target(ctrl_target_joint_pos[env_ids], env_ids=env_ids)

    def set_robot2_joint_target(self, joint_target, env_ids):
        joint_vel = torch.zeros_like(joint_target)
        self.raw.robot2_ctrl_target_joint_pos[env_ids, :] = joint_target
        with torch.inference_mode():
            self.raw._robot2.write_joint_state_to_sim(joint_target, joint_vel, env_ids=env_ids)
            self.raw._robot2.set_joint_position_target(self.raw.robot2_ctrl_target_joint_pos[env_ids], env_ids=env_ids)

    def robot2_solve_ik_joint_target(self, target_pos, target_quat):
        env_ids = self.env_ids()
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        pos_error, axis_angle_error = factory_control.get_pose_error(
            fingertip_midpoint_pos=self.raw.robot2_fingertip_midpoint_pos[env_ids],
            fingertip_midpoint_quat=self.raw.robot2_fingertip_midpoint_quat[env_ids],
            ctrl_target_fingertip_midpoint_pos=target_pos[env_ids],
            ctrl_target_fingertip_midpoint_quat=target_quat[env_ids],
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )
        delta_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
        delta_dof_pos = factory_control.get_delta_dof_pos(
            delta_pose=delta_pose,
            ik_method="dls",
            jacobian=self.raw.robot2_fingertip_midpoint_jacobian[env_ids],
            device=self.raw.device,
        )
        joint_pos = self.raw.robot2_joint_pos.clone()
        joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
        joint_pos[env_ids, 7:] = self.raw.robot2_ctrl_target_joint_pos[env_ids, 7:]
        return joint_pos[env_ids]

    def capture_peg_grasp_offset(self):
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        self._peg_grasp_offset_pos = self.raw.held_pos - self.raw.robot2_fingertip_midpoint_pos
        self._peg_grasp_quat = self.raw.held_quat.clone()

    def carry_peg_with_robot2(self):
        if self._peg_grasp_offset_pos is None or self._peg_grasp_quat is None:
            self.capture_peg_grasp_offset()
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        held_state = self.raw._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = (
            self.raw.robot2_fingertip_midpoint_pos + self._peg_grasp_offset_pos + self.raw.scene.env_origins
        )
        held_state[:, 3:7] = self._peg_grasp_quat
        held_state[:, 7:] = 0.0
        env_ids = self.env_ids()
        self.raw._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self.raw._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self.raw._held_asset.reset()

    def robot2_down_quat(self):
        euler = torch.tensor([3.14159, 0.0, 0.0], dtype=torch.float32, device=self.raw.device)
        euler = euler.unsqueeze(0).repeat(self.raw.num_envs, 1)
        return torch_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])

    def env_ids(self):
        return torch.arange(self.raw.num_envs, device=self.raw.device)

    def tensor_joint_pos(self, joint_pos):
        return torch.tensor(joint_pos, dtype=torch.float32, device=self.raw.device)

    def zero_action(self):
        return torch.zeros((self.raw.num_envs, 6), dtype=torch.float32, device=self.raw.device)

    def smoothstep(self, alpha: float):
        return alpha * alpha * (3.0 - 2.0 * alpha)

    def robot_handles(self, robot: str):
        if robot in ("franka1", "robot", "left"):
            return self.raw._robot, self.raw.ctrl_target_joint_pos
        if robot in ("franka2", "robot2", "right"):
            return self.raw._robot2, self.raw.robot2_ctrl_target_joint_pos
        raise ValueError(f"Unknown robot: {robot}")
