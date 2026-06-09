"""Atomic skills for the thin dual-Franka Factory peg-transfer rollout."""

from __future__ import annotations


class FactoryPegTransferAtomicSkills:
    """Reusable task-level primitives built on top of the motion mixin.

    These methods intentionally keep the successful thin script's low-level
    motion calls unchanged.  They only move task-specific flags, phase IDs, and
    success checks out of ``play_once`` so each skill is parameterized by robot
    identity and reusable geometric targets.
    """

    def insert_peg(self, obs, robot: str, hole_marker: str, max_steps: int):
        if not self.is_franka1(robot):
            raise NotImplementedError("The current checkpoint insertion skill is trained for franka1.")

        self._last_insert_hole_pos = self.marker_position(hole_marker, obs=obs)
        current_obs = obs
        for _ in range(max(1, int(max_steps))):
            self.apply_hold_targets()
            current_obs, _, _, _, actions = self.rl_skill.step(current_obs)
            self.apply_hold_targets()
            self.record(self.PHASE_INSERT, actions)
            if self.check_success(require=False):
                break
        return current_obs, self.check_success(require=False)

    def thread_nut(self, obs, robot: str, hole_marker: str, max_steps: int):
        if not self.is_franka1(robot):
            raise NotImplementedError("The current checkpoint threading skill is trained for franka1.")

        self._last_thread_hole_pos = self.marker_position(hole_marker, obs=obs)
        return self.run_checkpoint_skill(
            obs=obs,
            phase=self.PHASE_FRANKA1_THREAD,
            max_steps=max_steps,
            success_checker=self.check_threaded,
            stop_on_success=False,
        )

    def run_checkpoint_skill(self, obs, phase: int, max_steps: int, success_checker, stop_on_success: bool):
        current_obs = obs
        for _ in range(max(1, int(max_steps))):
            self.apply_hold_targets()
            current_obs, _, _, _, actions = self.rl_skill.step(current_obs)
            self.apply_hold_targets()
            self.record(phase, actions)
            if stop_on_success and success_checker(require=False):
                break
        return current_obs, success_checker(require=False)

    def release_lift(
        self,
        robot: str,
        lift_height: float,
        steps: int | None = None,
        gripper_width: float | None = None,
    ):
        articulation, ctrl_target = self.robot_handles(robot)
        env_ids = self.env_ids()
        width = self.default_release_width() if gripper_width is None else gripper_width
        phase = self.release_lift_phase(robot)

        current = articulation.data.joint_pos[env_ids].clone()
        open_target = current.clone()
        open_target[:, 7:] = width
        self.execute_joint_trajectory(
            articulation,
            open_target,
            ctrl_target,
            env_ids,
            self.gripper_steps,
            phase,
        )

        self.raw._compute_intermediate_values(self.raw.physics_dt)
        lift_pos, lift_quat = self.fingertip_pose(robot)
        lift_pos = lift_pos.clone()
        lift_pos[:, 2] += lift_height
        self.move_to_pose(
            robot,
            lift_pos,
            lift_quat,
            steps or self.release_lift_steps,
            phase,
            gripper_width=width,
        )
        self.hold_targets(self.hold_steps, phase)

    def return_home(self, robot: str, joint_pos=None, gripper_width: float | None = None, steps: int | None = None):
        articulation, ctrl_target = self.robot_handles(robot)
        env_ids = self.env_ids()
        width = self.default_release_width() if gripper_width is None else gripper_width
        home_joint_pos = self.home_joint_pos(robot) if joint_pos is None else joint_pos
        phase = self.home_phase(robot)

        home_target = articulation.data.joint_pos[env_ids].clone()
        home_target[:, 0:7] = self.tensor_joint_pos(home_joint_pos)
        home_target[:, 7:] = width
        self.execute_joint_trajectory(
            articulation,
            home_target,
            ctrl_target,
            env_ids,
            steps or self.home_steps,
            phase,
        )
        self.hold_targets(self.hold_steps, phase)

    def grasp(
        self,
        robot: str,
        target_marker: str,
        pregrasp_distance: float,
        move_steps: int,
        grasp_z_offset: float,
        target_quat=None,
        open_width: float = 0.08,
        closed_width: float = 0.0,
        preturn_turns: float = 0.0,
        preturn_steps: int = 0,
        preturn_direction: float = 1.0,
        preturn_phase: int | None = None,
    ):
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        target_pos = self.marker_position(target_marker)
        grasp_quat = self.default_down_quat(robot) if target_quat is None else target_quat
        pregrasp_phase, descend_phase, grasp_phase = self.grasp_phases(robot)

        pregrasp_pos = target_pos.clone()
        pregrasp_pos[:, 2] += pregrasp_distance
        grasp_pos = target_pos.clone()
        grasp_pos[:, 2] += grasp_z_offset

        self.move_to_pose(
            robot,
            pregrasp_pos,
            grasp_quat,
            move_steps,
            pregrasp_phase,
            gripper_width=open_width,
        )
        self.hold_targets(self.hold_steps, pregrasp_phase)

        descend_pos = pregrasp_pos.clone()
        descend_pos[:, 2] = grasp_pos[:, 2]
        self.move_to_pose(
            robot,
            descend_pos,
            grasp_quat,
            move_steps,
            descend_phase,
            gripper_width=open_width,
        )
        self.hold_targets(self.hold_steps, descend_phase)

        if abs(float(preturn_turns)) > 1e-6 and int(preturn_steps) > 0:
            phase = grasp_phase if preturn_phase is None else preturn_phase
            self.twist_wrist(
                robot=robot,
                turns=preturn_turns,
                steps=preturn_steps,
                phase=phase,
                gripper_width=open_width,
                direction=preturn_direction,
            )
            self.hold_targets(self.hold_steps, phase)

        self.set_gripper(robot, closed_width, self.gripper_steps, grasp_phase)
        if self.is_franka2(robot):
            self.capture_peg_grasp_offset()
        self.hold_targets(self.hold_steps, grasp_phase)
        return grasp_pos

    def unthread(
        self,
        robot: str,
        turns: float,
        steps: int,
        gripper_width: float,
        direction: float = -1.0,
    ):
        self.twist_wrist(
            robot=robot,
            turns=turns,
            steps=steps,
            phase=self.PHASE_FRANKA2_UNTHREAD,
            gripper_width=gripper_width,
            direction=direction,
        )

    def lift_vertical(
        self,
        robot: str,
        lift_height: float,
        steps: int,
        gripper_width: float,
        carry_peg: bool = False,
    ):
        phase = self.lift_phase(robot)
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        lift_pos, lift_quat = self.fingertip_pose(robot)
        lift_pos = lift_pos.clone()
        lift_pos[:, 2] += lift_height
        self.move_to_pose(
            robot,
            lift_pos,
            lift_quat,
            steps,
            phase,
            gripper_width=gripper_width,
            carry_peg=carry_peg,
        )
        self.hold_targets(self.hold_steps, phase, carry_peg=carry_peg)

    def twist_wrist(self, robot: str, turns: float, steps: int, phase: int, gripper_width: float, direction: float):
        if self.is_franka2(robot):
            self.robot2_reverse_twist_wrist_joint(
                turns=turns,
                steps=steps,
                phase=phase,
                gripper_width=gripper_width,
                direction=direction,
            )
            return
        raise NotImplementedError(f"Wrist twist is not implemented for {robot}.")

    def extract(
        self,
        robot: str,
        reference_marker: str,
        grasp_pos,
        pull_height: float,
        steps: int,
        target_quat=None,
    ):
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        reference_pos = self.marker_position(reference_marker)
        extract_quat = self.default_down_quat(robot) if target_quat is None else target_quat
        phase = self.extract_phase(robot)

        pull_pos = grasp_pos.clone()
        pull_pos[:, 0:2] = reference_pos[:, 0:2]
        pull_pos[:, 2] = grasp_pos[:, 2] + pull_height
        self.move_to_pose(
            robot,
            pull_pos,
            extract_quat,
            steps,
            phase,
            gripper_width=0.0,
            carry_peg=self.is_franka2(robot),
        )
        self.hold_targets(self.hold_steps, phase, carry_peg=self.is_franka2(robot))

    def marker_position(self, marker, obs=None):
        if callable(marker):
            return marker(self, obs)
        if hasattr(marker, "clone"):
            return marker.clone()
        if isinstance(obs, dict) and marker in obs and hasattr(obs[marker], "clone"):
            return obs[marker].clone()

        raw_attr = {
            "hole": "fixed_pos_obs_frame",
            "fixed": "fixed_pos_obs_frame",
            "fixed_pos": "fixed_pos_obs_frame",
            "peg": "held_pos",
            "held": "held_pos",
            "held_pos": "held_pos",
        }.get(marker, marker)
        if hasattr(self.raw, raw_attr):
            value = getattr(self.raw, raw_attr)
            if hasattr(value, "clone"):
                return value.clone()
        raise ValueError(f"Cannot resolve marker position: {marker}")

    def fingertip_pose(self, robot: str):
        if self.is_franka1(robot):
            return self.raw.fingertip_midpoint_pos.clone(), self.raw.fingertip_midpoint_quat.clone()
        if self.is_franka2(robot):
            return self.raw.robot2_fingertip_midpoint_pos.clone(), self.raw.robot2_fingertip_midpoint_quat.clone()
        raise ValueError(f"Unknown robot: {robot}")

    def move_to_pose(
        self,
        robot: str,
        target_pos,
        target_quat,
        steps: int,
        phase: int,
        gripper_width: float,
        carry_peg=False,
    ):
        if self.is_franka1(robot):
            if carry_peg:
                raise NotImplementedError("Peg carrying is currently implemented for franka2.")
            self.franka1_move_to_pose(target_pos, target_quat, steps, phase, gripper_width=gripper_width)
            return
        if self.is_franka2(robot):
            self.robot2_move_to_pose(
                target_pos,
                target_quat,
                steps,
                phase,
                gripper_width=gripper_width,
                carry_peg=carry_peg,
            )
            return
        raise ValueError(f"Unknown robot: {robot}")

    def set_gripper(self, robot: str, width: float, steps: int, phase: int):
        articulation, ctrl_target = self.robot_handles(robot)
        env_ids = self.env_ids()
        target = articulation.data.joint_pos[env_ids].clone()
        target[:, 7:] = width
        self.execute_joint_trajectory(articulation, target, ctrl_target, env_ids, steps, phase)

    def default_down_quat(self, robot: str):
        if self.is_franka2(robot):
            return self.robot2_down_quat()
        _, quat = self.fingertip_pose(robot)
        return quat

    def default_release_width(self) -> float:
        return self.raw.cfg_task.held_asset_cfg.diameter / 2 * 1.25

    def home_joint_pos(self, robot: str):
        if self.is_franka1(robot) and hasattr(self, "franka1_home_joint_pos"):
            return self.franka1_home_joint_pos
        if self.is_franka2(robot) and hasattr(self, "franka2_home_joint_pos"):
            return self.franka2_home_joint_pos
        raise ValueError(f"No home joint position configured for {robot}")

    def release_lift_phase(self, robot: str) -> int:
        if self.is_franka1(robot):
            return self.PHASE_FRANKA1_RELEASE_LIFT
        if self.is_franka2(robot):
            return self.PHASE_FRANKA2_EXTRACT
        raise ValueError(f"Unknown robot: {robot}")

    def home_phase(self, robot: str) -> int:
        if self.is_franka1(robot):
            return self.PHASE_FRANKA1_RELEASE_HOME
        if self.is_franka2(robot) and hasattr(self, "PHASE_FRANKA2_HOME"):
            return self.PHASE_FRANKA2_HOME
        if self.is_franka2(robot):
            return self.PHASE_FRANKA2_PREGRASP
        raise ValueError(f"Unknown robot: {robot}")

    def grasp_phases(self, robot: str):
        if self.is_franka2(robot):
            return self.PHASE_FRANKA2_PREGRASP, self.PHASE_FRANKA2_DESCEND, self.PHASE_FRANKA2_GRASP
        if self.is_franka1(robot):
            return self.PHASE_INSERT, self.PHASE_INSERT, self.PHASE_INSERT
        raise ValueError(f"Unknown robot: {robot}")

    def extract_phase(self, robot: str) -> int:
        if self.is_franka2(robot):
            return self.PHASE_FRANKA2_EXTRACT
        if self.is_franka1(robot):
            return self.PHASE_FRANKA1_RELEASE_LIFT
        raise ValueError(f"Unknown robot: {robot}")

    def lift_phase(self, robot: str) -> int:
        if self.is_franka2(robot) and hasattr(self, "PHASE_FRANKA2_LIFT"):
            return self.PHASE_FRANKA2_LIFT
        if self.is_franka1(robot):
            return self.PHASE_FRANKA1_RELEASE_LIFT
        raise ValueError(f"Unknown robot: {robot}")

    def is_franka1(self, robot: str) -> bool:
        return robot in ("franka1", "robot", "left")

    def is_franka2(self, robot: str) -> bool:
        return robot in ("franka2", "robot2", "right")
