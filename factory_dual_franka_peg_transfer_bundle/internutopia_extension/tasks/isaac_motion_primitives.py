"""Reusable IsaacLab motion primitives for synthetic manipulation tasks."""

from __future__ import annotations

import math

import torch

import isaacsim.core.utils.torch as torch_utils
from isaaclab_tasks.direct.factory import factory_control
from isaaclab_tasks.direct.factory import factory_utils


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

    def run_robot2_reverse_policy(self, phase: int, max_steps: int, gripper_width: float = 0.0, reverse_z: bool = True):
        actions = torch.zeros(
            (self.raw.num_envs, int(self.raw.cfg.action_space)), dtype=torch.float32, device=self.raw.device
        )
        obs = self.robot2_policy_observation(actions)
        for _ in range(max(1, int(max_steps))):
            with torch.inference_mode():
                agent_obs = self.agent.obs_to_torch(obs)
                actions = self.agent.get_action(agent_obs, is_deterministic=self.agent.is_deterministic)
            self.robot2_apply_policy_action(
                actions,
                reverse_yaw=True,
                reverse_z=reverse_z,
                gripper_width=gripper_width,
            )
            self.apply_hold_targets(skip_articulation=self.raw._robot2)
            self.raw.step_sim_no_action()
            self.record(phase, actions)
            obs = self.robot2_policy_observation(actions)
        self._robot2_hold_joint_pos = self.raw._robot2.data.joint_pos[self.env_ids()].clone()
        self.raw._robot2.set_joint_effort_target(torch.zeros_like(self.raw._robot2.data.joint_pos))

    def robot2_reverse_twist_wrist_joint(
        self,
        turns: float,
        steps: int,
        phase: int,
        gripper_width: float,
        direction: float = 1.0,
    ):
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        env_ids = self.env_ids()
        start_joint_pos = self.raw._robot2.data.joint_pos[env_ids].clone()
        target_joint_pos = start_joint_pos.clone()
        actions = torch.zeros(
            (self.raw.num_envs, int(self.raw.cfg.action_space)), dtype=torch.float32, device=self.raw.device
        )
        n_steps = max(1, int(steps))
        total_wrist = float(direction) * 2.0 * math.pi * float(turns)
        arm_kp = 120.0
        arm_kd = 2.5
        wrist_kp = 220.0
        wrist_kd = 4.0
        max_torque = 87.0

        for i in range(1, n_steps + 1):
            target_joint_pos[:, :] = start_joint_pos
            target_joint_pos[:, 6] = start_joint_pos[:, 6] + total_wrist * (i / n_steps)
            target_joint_pos[:, 7:9] = gripper_width
            self.raw.robot2_ctrl_target_joint_pos[env_ids, :] = target_joint_pos
            current_joint_pos = self.raw._robot2.data.joint_pos[env_ids].clone()
            current_joint_vel = self.raw._robot2.data.joint_vel[env_ids].clone()
            joint_torque = torch.zeros_like(current_joint_pos)
            joint_torque[:, 0:6] = arm_kp * (start_joint_pos[:, 0:6] - current_joint_pos[:, 0:6])
            joint_torque[:, 0:6] -= arm_kd * current_joint_vel[:, 0:6]
            joint_torque[:, 6] = wrist_kp * (target_joint_pos[:, 6] - current_joint_pos[:, 6])
            joint_torque[:, 6] -= wrist_kd * current_joint_vel[:, 6]
            joint_torque[:, 0:7] = torch.clamp(joint_torque[:, 0:7], -max_torque, max_torque)
            with torch.inference_mode():
                self.raw._robot2.set_joint_position_target(target_joint_pos, env_ids=env_ids)
                self.raw._robot2.set_joint_effort_target(joint_torque, env_ids=env_ids)
            self.apply_hold_targets(skip_articulation=self.raw._robot2)
            self.raw.step_sim_no_action()
            actions[:, 5] = total_wrist / max(1, n_steps)
            self.record(phase, actions)
            self.raw._compute_intermediate_values(self.raw.physics_dt)

        self._robot2_hold_joint_pos = self.raw._robot2.data.joint_pos[env_ids].clone()
        self.raw._robot2.set_joint_effort_target(torch.zeros_like(self.raw._robot2.data.joint_pos))

    def robot2_policy_observation(self, prev_actions):
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        noisy_fixed_pos = self.raw.fixed_pos_obs_frame + self.raw.init_fixed_pos_obs_noise
        obs_dict = {
            "fingertip_pos": self.raw.robot2_fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.raw.robot2_fingertip_midpoint_pos - noisy_fixed_pos,
            "fingertip_quat": self.raw.robot2_fingertip_midpoint_quat,
            "ee_linvel": self.raw.robot2_fingertip_midpoint_linvel,
            "ee_angvel": self.raw.robot2_fingertip_midpoint_angvel,
            "prev_actions": prev_actions,
        }
        return factory_utils.collapse_obs_dict(obs_dict, self.raw.cfg.obs_order + ["prev_actions"])

    def robot2_apply_policy_action(self, action, reverse_yaw: bool, reverse_z: bool, gripper_width: float):
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        if not hasattr(self, "_robot2_policy_actions"):
            self._robot2_policy_actions = torch.zeros_like(action)
        self._robot2_policy_actions = (
            self.raw.ema_factor * action.clone().to(self.raw.device)
            + (1.0 - self.raw.ema_factor) * self._robot2_policy_actions
        )

        pos_actions = self._robot2_policy_actions[:, 0:3] * self.raw.pos_threshold
        if reverse_z:
            pos_actions[:, 2] = torch.abs(pos_actions[:, 2])
        ctrl_target_fingertip_midpoint_pos = self.raw.robot2_fingertip_midpoint_pos + pos_actions

        rot_actions = self._robot2_policy_actions[:, 3:6].clone()
        if self.raw.cfg_task.unidirectional_rot:
            yaw_mag = (rot_actions[:, 2] + 1.0) * 0.5
            rot_actions[:, 2] = yaw_mag if reverse_yaw else -yaw_mag
        elif reverse_yaw:
            rot_actions[:, 2] = -rot_actions[:, 2]
        rot_actions = rot_actions * self.raw.rot_threshold

        fixed_pos_action_frame = self.raw.fixed_pos_obs_frame + self.raw.init_fixed_pos_obs_noise
        delta_pos = ctrl_target_fingertip_midpoint_pos - fixed_pos_action_frame
        pos_error_clipped = torch.clip(
            delta_pos,
            -self.raw.cfg.ctrl.pos_action_bounds[0],
            self.raw.cfg.ctrl.pos_action_bounds[1],
        )
        ctrl_target_fingertip_midpoint_pos = fixed_pos_action_frame + pos_error_clipped

        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.raw.device).repeat(self.raw.num_envs, 1),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(
            rot_actions_quat, self.raw.robot2_fingertip_midpoint_quat
        )

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0],
            pitch=target_euler_xyz[:, 1],
            yaw=target_euler_xyz[:, 2],
        )

        joint_torque, _ = factory_control.compute_dof_torque(
            cfg=self.raw.cfg,
            dof_pos=self.raw.robot2_joint_pos,
            dof_vel=self.raw.robot2_joint_vel,
            fingertip_midpoint_pos=self.raw.robot2_fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.raw.robot2_fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.raw.robot2_fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.raw.robot2_fingertip_midpoint_angvel,
            jacobian=self.raw.robot2_fingertip_midpoint_jacobian,
            arm_mass_matrix=self.raw.robot2_arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.raw.task_prop_gains,
            task_deriv_gains=self.raw.task_deriv_gains,
            device=self.raw.device,
            dead_zone_thresholds=self.raw.dead_zone_thresholds,
        )
        self.raw.robot2_ctrl_target_joint_pos[:, 7:9] = gripper_width
        joint_torque[:, 7:9] = 0.0
        self.raw._robot2.set_joint_position_target(self.raw.robot2_ctrl_target_joint_pos)
        self.raw._robot2.set_joint_effort_target(joint_torque)

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
        preturn_turns: float = 0.0,
        preturn_steps: int = 0,
        preturn_direction: float = 1.0,
        preturn_phase: int | None = None,
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

        if abs(float(preturn_turns)) > 1e-6 and int(preturn_steps) > 0:
            self.robot2_reverse_twist_wrist_joint(
                turns=preturn_turns,
                steps=preturn_steps,
                phase=grasp_phase if preturn_phase is None else preturn_phase,
                gripper_width=open_width,
                direction=preturn_direction,
            )
            self.hold_targets(self.hold_steps, grasp_phase if preturn_phase is None else preturn_phase)

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

    def robot2_twist_tcp_about_world_z(self, turns: float, steps: int, phase: int, gripper_width: float, direction: float):
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        target_pos = self.raw.robot2_fingertip_midpoint_pos.clone()
        env_ids = self.env_ids()
        total_steps = max(1, int(steps))
        step_yaw = float(direction) * 2.0 * math.pi * float(turns) / total_steps
        step_yaw_tensor = torch.full((self.raw.num_envs,), step_yaw, dtype=torch.float32, device=self.raw.device)
        zeros = torch.zeros_like(step_yaw_tensor)

        for _ in range(total_steps):
            self.raw._compute_intermediate_values(self.raw.physics_dt)
            world_delta_quat = torch_utils.quat_from_euler_xyz(zeros, zeros, step_yaw_tensor)
            target_quat = torch_utils.quat_mul(world_delta_quat, self.raw.robot2_fingertip_midpoint_quat)
            joint_target = self.robot2_solve_ik_joint_target(target_pos, target_quat)
            joint_target[:, 7:] = gripper_width
            self.set_robot2_joint_target(joint_target, env_ids)
            self.apply_hold_targets(skip_articulation=self.raw._robot2)
            self.raw.step_sim_no_action()
            self.record(phase, self.zero_action())
        self._robot2_hold_joint_pos = self.raw._robot2.data.joint_pos[env_ids].clone()

    def franka1_move_to_pose(self, target_pos, target_quat, steps: int, phase: int, gripper_width: float):
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        start = self.raw.fingertip_midpoint_pos.clone()
        total_steps = max(1, int(steps) * max(1, int(self.robot2_ik_substeps)))
        env_ids = self.env_ids()
        for i in range(1, total_steps + 1):
            alpha = i / total_steps
            waypoint = torch.lerp(start, target_pos, self.smoothstep(alpha))
            joint_target = self.franka1_solve_ik_joint_target(waypoint, target_quat)
            joint_target[:, 7:] = gripper_width
            self.set_franka1_joint_target(joint_target, env_ids)
            self.apply_hold_targets(skip_articulation=self.raw._robot)
            self.raw.step_sim_no_action()
            self.record(phase, self.zero_action())
        self._franka1_hold_joint_pos = self.raw._robot.data.joint_pos[env_ids].clone()

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

    def set_franka1_joint_target(self, joint_target, env_ids):
        joint_vel = torch.zeros_like(joint_target)
        self.raw.ctrl_target_joint_pos[env_ids, :] = joint_target
        with torch.inference_mode():
            self.raw._robot.write_joint_state_to_sim(joint_target, joint_vel, env_ids=env_ids)
            self.raw._robot.set_joint_position_target(self.raw.ctrl_target_joint_pos[env_ids], env_ids=env_ids)

    def franka1_solve_ik_joint_target(self, target_pos, target_quat):
        env_ids = self.env_ids()
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        pos_error, axis_angle_error = factory_control.get_pose_error(
            fingertip_midpoint_pos=self.raw.fingertip_midpoint_pos[env_ids],
            fingertip_midpoint_quat=self.raw.fingertip_midpoint_quat[env_ids],
            ctrl_target_fingertip_midpoint_pos=target_pos[env_ids],
            ctrl_target_fingertip_midpoint_quat=target_quat[env_ids],
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )
        delta_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
        delta_dof_pos = factory_control.get_delta_dof_pos(
            delta_pose=delta_pose,
            ik_method="dls",
            jacobian=self.raw.fingertip_midpoint_jacobian[env_ids],
            device=self.raw.device,
        )
        joint_pos = self.raw.joint_pos.clone()
        joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
        joint_pos[env_ids, 7:] = self.raw.ctrl_target_joint_pos[env_ids, 7:]
        return joint_pos[env_ids]

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
