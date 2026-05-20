"""Thin RobotWin-style dual-Franka Factory peg-transfer task."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ISAACLAB_ROOT = os.path.join(_PROJECT_ROOT, "IsaacLab")
for _rel_path in (
    "source/isaaclab",
    "source/isaaclab_assets",
    "source/isaaclab_mimic",
    "source/isaaclab_rl",
    "source/isaaclab_tasks",
):
    _path = os.path.join(_ISAACLAB_ROOT, _rel_path)
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import isaacsim

    if getattr(isaacsim, "__file__", None) is None and os.environ.get("ISAAC_PATH"):
        isaacsim.__file__ = os.path.join(os.environ["ISAAC_PATH"], "python", "isaacsim", "__init__.py")
    if not hasattr(isaacsim, "SimulationApp"):
        from isaacsim.simulation_app import SimulationApp

        isaacsim.SimulationApp = SimulationApp
except ImportError:
    pass

from isaaclab.app import AppLauncher


DEFAULT_CHECKPOINT = os.path.join(_PROJECT_ROOT, "checkpoints", "Factory", "test", "nn", "Factory.pth")
DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "factory_dual_franka_peg_transfer_thin")
FRANKA_RESET_JOINT_POS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
ROBOT2_INITIAL_JOINT_POS = (0.0015178, -0.19651, -0.0014364, -1.9761, -0.00027717, 1.7796, 0.78556)
ROBOT2_GRASP_Z_OFFSET = 0.025


parser = argparse.ArgumentParser(description="Thin dual-Franka Factory peg transfer rollout.")
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Franka1 RL-Games checkpoint path.")
parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for rollout HDF5/MP4 files.")
parser.add_argument("--num_envs", type=int, default=1, help="Only one env is saved; keep this at 1 for collection.")
parser.add_argument("--seed", type=int, default=0, help="Environment seed. Use -1 for random.")
parser.add_argument("--insert_steps", type=int, default=260, help="Maximum Franka1 RL insertion steps.")
parser.add_argument("--extract_steps", type=int, default=36, help="IK waypoints used by Franka2 for extraction.")
parser.add_argument("--home_steps", type=int, default=120, help="Interpolated steps for Franka1 to return home.")
parser.add_argument("--gripper_steps", type=int, default=30, help="Interpolated steps for scripted gripper actions.")
parser.add_argument("--robot2_ik_substeps", type=int, default=8, help="Joint interpolation substeps for Franka2 IK.")
parser.add_argument("--hold_steps", type=int, default=12, help="Extra steps to keep a target after reaching it.")
parser.add_argument("--approach_height", type=float, default=0.12, help="Franka2 approach height above the peg.")
parser.add_argument("--pull_height", type=float, default=0.20, help="Franka2 upward pull distance after grasp.")
parser.add_argument("--stop_on_success", action="store_true", default=True, help="Stop insertion once success is true.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
from internutopia_extension.tasks.synthetic_base_task import add_synthetic_data_args

add_synthetic_data_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.num_envs != 1:
    raise ValueError("This dual-Franka task records one rollout at a time; run with --num_envs 1.")
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
from pxr import Gf, UsdGeom

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from internutopia_extension.tasks.factory_dual_franka_env import (
    DUAL_TASK_ID,
    configure_factory_dual_franka_env_cfg,
    register_dual_franka_factory_env,
)
from internutopia_extension.tasks.isaac_motion_primitives import IsaacDualFrankaMotionMixin
from internutopia_extension.tasks.synthetic_base_task import (
    SyntheticDataBaseTask,
    configure_viewer,
    create_rl_games_checkpoint_skill,
)


register_dual_franka_factory_env()


class DualFrankaPegTransferTask(IsaacDualFrankaMotionMixin, SyntheticDataBaseTask):
    PHASE_INSERT = 0
    PHASE_FRANKA1_RELEASE_HOME = 1
    PHASE_FRANKA2_PREGRASP = 2
    PHASE_FRANKA2_DESCEND = 3
    PHASE_FRANKA2_GRASP = 4
    PHASE_FRANKA2_EXTRACT = 5

    def __init__(
        self,
        rl_skill,
        insert_steps: int,
        extract_steps: int,
        approach_height: float,
        pull_height: float,
        home_steps: int,
        gripper_steps: int,
        robot2_ik_substeps: int,
        hold_steps: int,
        stop_on_success: bool,
        **base_kwargs,
    ):
        super().__init__(rl_skill=rl_skill, **base_kwargs)
        self.insert_steps = insert_steps
        self.extract_steps = extract_steps
        self.approach_height = approach_height
        self.pull_height = pull_height
        self.home_steps = home_steps
        self.gripper_steps = gripper_steps
        self.robot2_ik_substeps = robot2_ik_substeps
        self.hold_steps = hold_steps
        self.stop_on_success = stop_on_success
        self.table_length = None
        self._peg_grasp_offset_pos = None
        self._peg_grasp_quat = None
        self._franka1_hold_joint_pos = None
        self._robot2_hold_joint_pos = None

    def play_once(self):
        obs = self.reset_scene()
        self.table_length = self.measure_table_length()
        self.prepare_robot2_home_hold(ROBOT2_INITIAL_JOINT_POS)

        self.run_checkpoint_policy_with_holds(
            obs=obs,
            phase=self.PHASE_INSERT,
            max_steps=self.insert_steps,
            stop_on_success=self.stop_on_success,
        )
        inserted = self.check_success(require=False)
        if inserted:
            self.open_gripper_and_move_joints(
                robot="franka1",
                joint_pos=FRANKA_RESET_JOINT_POS,
                gripper_width=self.raw.cfg_task.held_asset_cfg.diameter / 2 * 1.25,
                steps=self.home_steps,
                phase=self.PHASE_FRANKA1_RELEASE_HOME,
            )
            self.franka2_grasp_and_extract()

        success = inserted and self.check_extracted()
        dataset_path, video_path = self.finish_episode(success)
        self.info["info"] = {
            "{A}": "IsaacLab/Factory/DualFrankaPegTransfer",
            "{a}": "franka1",
            "{b}": "franka2",
            "checkpoint": self.resume_path,
            "dataset": dataset_path,
            "video": video_path,
            "inserted": inserted,
            "success": success,
            "table_length": self.table_length,
            "phase_names": self.phase_names(),
        }
        return self.info

    def franka2_grasp_and_extract(self):
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        peg_pos = self.raw.held_pos.clone()
        hole_pos = self.raw.fixed_pos_obs_frame.clone()
        down_quat = self.robot2_down_quat()
        grasp_pos = self.robot2_grasp_actor(
            peg_pos=peg_pos,
            target_quat=down_quat,
            pregrasp_height=self.approach_height,
            grasp_z_offset=ROBOT2_GRASP_Z_OFFSET,
            move_steps=self.extract_steps,
            pregrasp_phase=self.PHASE_FRANKA2_PREGRASP,
            descend_phase=self.PHASE_FRANKA2_DESCEND,
            grasp_phase=self.PHASE_FRANKA2_GRASP,
        )
        self.robot2_pull_actor(
            hole_pos=hole_pos,
            grasp_pos=grasp_pos,
            target_quat=down_quat,
            pull_height=self.pull_height,
            steps=self.extract_steps,
            phase=self.PHASE_FRANKA2_EXTRACT,
        )

    def check_success(self, require: bool = True) -> bool:
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        success = self.raw._get_curr_successes(self.raw.cfg_task.success_threshold, check_rot=False)
        ok = bool(success[0].item())
        if require and not ok:
            print("[WARN] Franka1 insertion is not successful yet.", flush=True)
        return ok

    def check_extracted(self) -> bool:
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        return bool((self.raw.held_pos[0, 2] - self.raw.fixed_pos_obs_frame[0, 2]).item() > 0.08)

    def finish_episode(self, success: bool):
        return self.save_outputs(
            success=success,
            stem_prefix="factory_dual_franka_peg_transfer_thin",
            phase_names=self.phase_names(),
            metadata={
                "insert_steps": self.insert_steps,
                "extract_steps": self.extract_steps,
                "approach_height": self.approach_height,
                "pull_height": self.pull_height,
                "home_steps": self.home_steps,
                "gripper_steps": self.gripper_steps,
                "robot2_ik_substeps": self.robot2_ik_substeps,
                "hold_steps": self.hold_steps,
                "table_length": self.table_length,
            },
        )

    def measure_table_length(self) -> float | None:
        stage = self.raw.sim.stage
        prim = stage.GetPrimAtPath("/World/envs/env_0/Table")
        if not prim.IsValid():
            return None
        bbox_cache = UsdGeom.BBoxCache(UsdGeom.GetStageMetersPerUnit(stage), [UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        size: Gf.Vec3d = bbox.GetSize()
        min_pos: Gf.Vec3d = bbox.GetMin()
        max_pos: Gf.Vec3d = bbox.GetMax()
        length = float(max(size[0], size[1]))
        print(
            "[INFO] desk005 bbox "
            f"size: x={float(size[0]):.3f}, y={float(size[1]):.3f}, z={float(size[2]):.3f}; "
            f"z_min={float(min_pos[2]):.3f}, z_max={float(max_pos[2]):.3f}",
            flush=True,
        )
        return length

    def phase_names(self) -> dict[str, str]:
        return {
            str(self.PHASE_INSERT): "franka1_insert_policy",
            str(self.PHASE_FRANKA1_RELEASE_HOME): "franka1_release_and_home",
            str(self.PHASE_FRANKA2_PREGRASP): "franka2_move_to_pregrasp",
            str(self.PHASE_FRANKA2_DESCEND): "franka2_vertical_descend_to_grasp",
            str(self.PHASE_FRANKA2_GRASP): "franka2_grasp_peg",
            str(self.PHASE_FRANKA2_EXTRACT): "franka2_extract_peg",
        }


@hydra_task_config(DUAL_TASK_ID, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    configure_factory_dual_franka_env_cfg(env_cfg, args_cli)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    configure_viewer(env_cfg, args_cli)
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device
    agent_cfg["params"]["seed"] = args_cli.seed

    rl_skill = create_rl_games_checkpoint_skill(
        task_name=DUAL_TASK_ID,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        checkpoint=args_cli.checkpoint,
    )
    task = DualFrankaPegTransferTask(
        rl_skill=rl_skill,
        insert_steps=args_cli.insert_steps,
        extract_steps=args_cli.extract_steps,
        approach_height=args_cli.approach_height,
        pull_height=args_cli.pull_height,
        home_steps=args_cli.home_steps,
        gripper_steps=args_cli.gripper_steps,
        robot2_ik_substeps=args_cli.robot2_ik_substeps,
        hold_steps=args_cli.hold_steps,
        stop_on_success=args_cli.stop_on_success,
        task_name=DUAL_TASK_ID,
        output_dir=args_cli.output_dir,
        video_fps=args_cli.video_fps,
        video_frame_repeat=args_cli.video_frame_repeat,
        viewer_eye=args_cli.viewer_eye,
        viewer_lookat=args_cli.viewer_lookat,
    )
    info = task.play_once()
    print(json.dumps(info, indent=2, sort_keys=True), flush=True)
    rl_skill.env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
