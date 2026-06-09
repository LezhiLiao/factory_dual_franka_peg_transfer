"""Thin RobotWin-style dual-Franka Factory nut-transfer task."""

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


DEFAULT_CHECKPOINT = "/data/user/InternUtopia/IsaacLab/logs/rl_games/Factory/test/nn/Factory.pth"
DEFAULT_OUTPUT_DIR = "/data/user/InternUtopia/outputs/factory_dual_franka_nut_transfer_thin"
FRANKA_RESET_JOINT_POS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
ROBOT2_INITIAL_JOINT_POS = (0.0015178, -0.19651, -0.0014364, -1.9761, -0.00027717, 1.7796, 0.78556)
ROBOT2_NUT_GRASP_Z_OFFSET = 0.008
ROBOT2_NUT_GRASP_Z_ADJUSTMENT = -0.005
ROBOT2_NUT_CLOSED_GRIPPER_WIDTH = 0.01
WRIST_CAMERA_LOCAL_OFFSET = (0.05, 0.0, -0.02)


parser = argparse.ArgumentParser(description="Thin dual-Franka Factory nut transfer rollout.")
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Factory NutThread RL-Games checkpoint path.")
parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for rollout HDF5/MP4 files.")
parser.add_argument("--num_envs", type=int, default=1, help="Only one env is saved; keep this at 1 for collection.")
parser.add_argument("--seed", type=int, default=0, help="Environment seed. Use -1 for random.")
parser.add_argument("--thread_steps", type=int, default=390, help="Franka1 RL nut-threading steps, roughly three turns.")
parser.add_argument("--unthread_steps", type=int, default=480, help="Franka2 wrist-joint reverse-twist steps.")
parser.add_argument("--robot2_unthread_turns", type=float, default=0.85, help="Franka2 panda_joint7 reverse turns while keeping other joints fixed.")
parser.add_argument("--robot2_preturn_steps", type=int, default=320, help="Open-gripper wrist preturn steps before grasping.")
parser.add_argument("--robot2_preturn_turns", type=float, default=0.85, help="Open-gripper wrist preturns before closing the gripper.")
parser.add_argument("--robot2_move_steps", type=int, default=36, help="IK waypoints used by Franka2 for pregrasp/lift.")
parser.add_argument("--home_steps", type=int, default=120, help="Interpolated steps for Franka1 to return home.")
parser.add_argument("--gripper_steps", type=int, default=30, help="Interpolated steps for scripted gripper actions.")
parser.add_argument("--robot2_ik_substeps", type=int, default=8, help="Joint interpolation substeps for Franka2 IK.")
parser.add_argument("--hold_steps", type=int, default=12, help="Extra steps to keep a target after reaching it.")
parser.add_argument("--approach_height", type=float, default=0.12, help="Franka2 approach height above the nut.")
parser.add_argument("--lift_height", type=float, default=0.20, help="Franka2 upward lift distance after unthreading.")
parser.add_argument("--robot2_grasp_z_offset", type=float, default=ROBOT2_NUT_GRASP_Z_OFFSET, help="Franka2 grasp height offset above nut root.")
parser.add_argument("--robot2_closed_gripper_width", type=float, default=ROBOT2_NUT_CLOSED_GRIPPER_WIDTH, help="Deprecated; Franka2 closed gripper width is fixed at 0.01.")
parser.add_argument("--release_lift_height", type=float, default=0.07, help="Franka1 vertical lift distance before reset.")
parser.add_argument("--release_lift_steps", type=int, default=36, help="IK waypoints for Franka1 vertical lift before reset.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument(
    "--wrist_camera_translate",
    type=float,
    nargs=3,
    default=(0.05, 0.0, -0.02),
    help="D455 pose translation under panda_hand.",
)
parser.add_argument(
    "--wrist_camera_rotate_zyx",
    type=float,
    nargs=3,
    default=(0.0, 90.0, 180.0),
    help="D455 pose rotationZYX under panda_hand, in degrees.",
)
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
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils.hydra import hydra_task_config
from internutopia_extension.tasks.factory_dual_franka_env import (
    DUAL_NUT_THREAD_TASK_ID,
    configure_factory_dual_franka_env_cfg,
    register_dual_franka_factory_nut_env,
)
from internutopia_extension.tasks.isaac_motion_primitives import IsaacDualFrankaMotionMixin
from internutopia_extension.tasks.synthetic_base_task import (
    SyntheticDataBaseTask,
    configure_viewer,
    create_rl_games_checkpoint_skill,
)
from factory_dual_franka_peg_transfer_atomic_skills import FactoryPegTransferAtomicSkills


register_dual_franka_factory_nut_env()


class DualFrankaNutTransferTask(FactoryPegTransferAtomicSkills, IsaacDualFrankaMotionMixin, SyntheticDataBaseTask):
    PHASE_FRANKA1_THREAD = 0
    PHASE_FRANKA1_RELEASE_HOME = 1
    PHASE_FRANKA1_RELEASE_LIFT = 2
    PHASE_FRANKA2_PREGRASP = 3
    PHASE_FRANKA2_DESCEND = 4
    PHASE_FRANKA2_PRETURN = 5
    PHASE_FRANKA2_GRASP = 6
    PHASE_FRANKA2_UNTHREAD = 7
    PHASE_FRANKA2_LIFT = 8
    PHASE_FRANKA2_HOME = 9

    def __init__(
        self,
        rl_skill,
        thread_steps: int,
        unthread_steps: int,
        robot2_unthread_turns: float,
        robot2_preturn_steps: int,
        robot2_preturn_turns: float,
        robot2_move_steps: int,
        approach_height: float,
        lift_height: float,
        robot2_grasp_z_offset: float,
        robot2_closed_gripper_width: float,
        release_lift_height: float,
        release_lift_steps: int,
        home_steps: int,
        gripper_steps: int,
        robot2_ik_substeps: int,
        hold_steps: int,
        **base_kwargs,
    ):
        super().__init__(rl_skill=rl_skill, **base_kwargs)
        self.thread_steps = thread_steps
        self.unthread_steps = unthread_steps
        self.robot2_unthread_turns = robot2_unthread_turns
        self.robot2_preturn_steps = robot2_preturn_steps
        self.robot2_preturn_turns = robot2_preturn_turns
        self.robot2_move_steps = robot2_move_steps
        self.approach_height = approach_height
        self.lift_height = lift_height
        self.robot2_grasp_z_offset = robot2_grasp_z_offset + ROBOT2_NUT_GRASP_Z_ADJUSTMENT
        self.robot2_closed_gripper_width = robot2_closed_gripper_width
        self.release_lift_height = release_lift_height
        self.release_lift_steps = release_lift_steps
        self.home_steps = home_steps
        self.gripper_steps = gripper_steps
        self.robot2_ik_substeps = robot2_ik_substeps
        self.hold_steps = hold_steps
        self.table_length = None
        self._peg_grasp_offset_pos = None
        self._peg_grasp_quat = None
        self._franka1_hold_joint_pos = None
        self._robot2_hold_joint_pos = None
        self.franka1_home_joint_pos = FRANKA_RESET_JOINT_POS
        self.franka2_home_joint_pos = ROBOT2_INITIAL_JOINT_POS

    def play_once(self):
        obs = self.reset_scene()
        self.table_length = self.measure_table_length()
        self.prepare_robot2_home_hold(ROBOT2_INITIAL_JOINT_POS)

        _, threaded = self.thread_nut(
            obs=obs,
            robot="franka1",
            hole_marker="hole",
            max_steps=self.thread_steps,
        )
        self.release_lift(
            robot="franka1",
            lift_height=self.release_lift_height,
        )
        self.return_home(robot="franka1")
        self.grasp(
            robot="franka2",
            target_marker="held_pos",
            pregrasp_distance=self.approach_height,
            move_steps=self.robot2_move_steps,
            grasp_z_offset=self.robot2_grasp_z_offset,
            closed_width=self.robot2_closed_gripper_width,
            preturn_turns=self.robot2_preturn_turns,
            preturn_steps=self.robot2_preturn_steps,
            preturn_direction=1.0,
            preturn_phase=self.PHASE_FRANKA2_PRETURN,
        )
        self.unthread(
            robot="franka2",
            turns=self.robot2_unthread_turns,
            steps=self.unthread_steps,
            gripper_width=self.robot2_closed_gripper_width,
            direction=-1.0,
        )
        self.lift_vertical(
            robot="franka2",
            lift_height=self.lift_height,
            steps=self.robot2_move_steps,
            gripper_width=self.robot2_closed_gripper_width,
        )
        self.return_home(
            robot="franka2",
            gripper_width=self.robot2_closed_gripper_width,
        )

        unthreaded = self.check_unthreaded()
        success = threaded and unthreaded
        dataset_path, video_path = self.finish_episode(success)
        self.info["info"] = {
            "{A}": "IsaacLab/Factory/DualFrankaNutTransfer",
            "{a}": "franka1",
            "{b}": "franka2",
            "checkpoint": self.resume_path,
            "dataset": dataset_path,
            "video": video_path,
            "threaded": threaded,
            "unthreaded": unthreaded,
            "success": success,
            "table_length": self.table_length,
            "phase_names": self.phase_names(),
        }
        return self.info

    def check_threaded(self, require: bool = True) -> bool:
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        success = self.raw._get_curr_successes(self.raw.cfg_task.success_threshold, check_rot=True)
        ok = bool(success[0].item())
        if require and not ok:
            print("[WARN] Franka1 nut threading is not successful yet.", flush=True)
        return ok

    def check_success(self, require: bool = True) -> bool:
        return self.check_threaded(require=require)

    def check_unthreaded(self) -> bool:
        self.raw._compute_intermediate_values(self.raw.physics_dt)
        return bool((self.raw.held_pos[0, 2] - self.raw.fixed_pos_obs_frame[0, 2]).item() > 0.08)

    def prepare_render_camera(self):
        if self.record_camera in ("viewer", "franka1_d455", "franka2_d455"):
            return

        self.raw._compute_intermediate_values(self.raw.physics_dt)
        env_origin = self.raw.scene.env_origins[0]
        offset = torch.tensor(WRIST_CAMERA_LOCAL_OFFSET, dtype=torch.float32, device=self.raw.device)

        if self.record_camera == "franka2_wrist_viewer":
            wrist_pos = self.raw.robot2_fingertip_midpoint_pos[0] + env_origin
            wrist_quat = self.raw.robot2_fingertip_midpoint_quat[0]
            target_pos = self.raw.held_pos[0] + env_origin
        else:
            wrist_pos = self.raw.fingertip_midpoint_pos[0] + env_origin
            wrist_quat = self.raw.fingertip_midpoint_quat[0]
            target_pos = self.raw.fixed_pos_obs_frame[0] + env_origin

        eye = wrist_pos + quat_apply(wrist_quat.unsqueeze(0), offset.unsqueeze(0)).squeeze(0)
        target = target_pos
        if torch.linalg.norm(target - eye).item() < 0.03:
            forward = torch.tensor((0.20, 0.0, 0.0), dtype=torch.float32, device=self.raw.device)
            target = eye + quat_apply(wrist_quat.unsqueeze(0), forward.unsqueeze(0)).squeeze(0)

        self.raw.sim.set_camera_view(
            eye=tuple(float(v) for v in eye.detach().cpu().tolist()),
            target=tuple(float(v) for v in target.detach().cpu().tolist()),
        )

    def finish_episode(self, success: bool):
        return self.save_outputs(
            success=success,
            stem_prefix="factory_dual_franka_nut_transfer_thin",
            phase_names=self.phase_names(),
            metadata={
                "thread_steps": self.thread_steps,
                "unthread_steps": self.unthread_steps,
                "robot2_move_steps": self.robot2_move_steps,
                "approach_height": self.approach_height,
                "lift_height": self.lift_height,
                "robot2_grasp_z_offset": self.robot2_grasp_z_offset,
                "release_lift_height": self.release_lift_height,
                "release_lift_steps": self.release_lift_steps,
                "home_steps": self.home_steps,
                "gripper_steps": self.gripper_steps,
                "robot2_ik_substeps": self.robot2_ik_substeps,
                "hold_steps": self.hold_steps,
                "table_length": self.table_length,
                "wrist_camera_model": "Intel RealSense D455 USD asset",
                "wrist_camera_source": args_cli.record_camera,
                "wrist_camera_translate": list(args_cli.wrist_camera_translate),
                "wrist_camera_rotate_zyx": list(args_cli.wrist_camera_rotate_zyx),
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
            str(self.PHASE_FRANKA1_THREAD): "franka1_thread_policy",
            str(self.PHASE_FRANKA1_RELEASE_HOME): "franka1_release_and_home",
            str(self.PHASE_FRANKA1_RELEASE_LIFT): "franka1_release_vertical_lift",
            str(self.PHASE_FRANKA2_PREGRASP): "franka2_move_to_pregrasp",
            str(self.PHASE_FRANKA2_DESCEND): "franka2_vertical_descend_to_grasp",
            str(self.PHASE_FRANKA2_PRETURN): "franka2_open_gripper_wrist_preturn",
            str(self.PHASE_FRANKA2_GRASP): "franka2_grasp_nut",
            str(self.PHASE_FRANKA2_UNTHREAD): "franka2_wrist_joint_reverse_twist",
            str(self.PHASE_FRANKA2_LIFT): "franka2_lift_nut",
            str(self.PHASE_FRANKA2_HOME): "franka2_home_after_lift",
        }


@hydra_task_config(DUAL_NUT_THREAD_TASK_ID, "rl_games_cfg_entry_point")
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
        task_name=DUAL_NUT_THREAD_TASK_ID,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        checkpoint=args_cli.checkpoint,
    )
    task = DualFrankaNutTransferTask(
        rl_skill=rl_skill,
        thread_steps=args_cli.thread_steps,
        unthread_steps=args_cli.unthread_steps,
        robot2_unthread_turns=args_cli.robot2_unthread_turns,
        robot2_preturn_steps=args_cli.robot2_preturn_steps,
        robot2_preturn_turns=args_cli.robot2_preturn_turns,
        robot2_move_steps=args_cli.robot2_move_steps,
        approach_height=args_cli.approach_height,
        lift_height=args_cli.lift_height,
        robot2_grasp_z_offset=args_cli.robot2_grasp_z_offset,
        robot2_closed_gripper_width=ROBOT2_NUT_CLOSED_GRIPPER_WIDTH,
        release_lift_height=args_cli.release_lift_height,
        release_lift_steps=args_cli.release_lift_steps,
        home_steps=args_cli.home_steps,
        gripper_steps=args_cli.gripper_steps,
        robot2_ik_substeps=args_cli.robot2_ik_substeps,
        hold_steps=args_cli.hold_steps,
        task_name=DUAL_NUT_THREAD_TASK_ID,
        output_dir=args_cli.output_dir,
        video_fps=args_cli.video_fps,
        video_frame_repeat=args_cli.video_frame_repeat,
        record_camera=args_cli.record_camera,
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
