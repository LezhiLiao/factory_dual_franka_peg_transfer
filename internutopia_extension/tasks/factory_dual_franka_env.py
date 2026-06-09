"""Dual-Franka IsaacLab Factory scene used by synthetic peg-transfer tasks."""

from __future__ import annotations

import copy
import os

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab_tasks.direct.factory import agents, factory_utils
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskNutThreadCfg, FactoryTaskPegInsertCfg


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DESK005_USD = os.path.join(PROJECT_ROOT, "asset", "desk005", "model_table_7.usd")
FRANKA_CAMERA_USD = os.path.join(PROJECT_ROOT, "asset", "franka", "franka_mimic_fixed_camera.usda")
REALSENSE_USD = os.path.join(
    PROJECT_ROOT,
    "asset",
    "ur5e_robotiq",
    "ur5e_robotiq_2f85_fixed_camera",
    "SubUSDs",
    "rsd455.usd",
)
GROUND_Z = -1.05
DESK005_TABLE_HEIGHT = 0.744
DESK005_TABLE_BASE_Z = GROUND_Z
DESK005_TABLETOP_Z = DESK005_TABLE_BASE_Z + DESK005_TABLE_HEIGHT
TASK_X_OFFSET = -0.2
DUAL_TASK_ID = "InternUtopia-Factory-DualFrankaPegTransfer-Direct-v0"
DUAL_NUT_THREAD_TASK_ID = "InternUtopia-Factory-DualFrankaNutTransfer-Direct-v0"
ROBOT2_INITIAL_JOINT_POS = (0.0015178, -0.19651, -0.0014364, -1.9761, -0.00027717, 1.7796, 0.78556)
D455_COLOR_CAMERA_REL_PATH = "panda_hand/realsense/RSD455/Camera_OmniVision_OV9782_Color"


def _usd_path(path: str) -> str:
    return path.replace("\\", "/")


def _ensure_franka_camera_usd(
    base_usd_path: str,
    camera_translate: tuple[float, float, float],
    camera_rotate_zyx: tuple[float, float, float],
) -> str:
    os.makedirs(os.path.dirname(FRANKA_CAMERA_USD), exist_ok=True)
    tx, ty, tz = camera_translate
    rz, ry, rx = camera_rotate_zyx
    usd_content = f"""#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Root" (
    prepend payload = @{_usd_path(base_usd_path)}@
)
{{
    over "panda_hand"
    {{
        def Xform "realsense" (
            prepend payload = @{_usd_path(REALSENSE_USD)}@
        )
        {{
            double3 xformOp:rotateZYX = ({rz}, {ry}, {rx})
            double3 xformOp:translate = ({tx}, {ty}, {tz})
            double3 xformOp:scale = (1, 1, 1)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZYX", "xformOp:scale"]

            over "RSD455" (
                delete apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
            )
            {{
            }}
        }}
    }}
}}
"""
    try:
        with open(FRANKA_CAMERA_USD, "r", encoding="utf-8") as f:
            if f.read() == usd_content:
                return FRANKA_CAMERA_USD
    except FileNotFoundError:
        pass
    with open(FRANKA_CAMERA_USD, "w", encoding="utf-8") as f:
        f.write(usd_content)
    return FRANKA_CAMERA_USD


class DualFrankaFactoryEnv(FactoryEnv):
    """Factory peg-in-hole env with a second scripted Franka across desk005."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        factory_utils.set_body_inertias(self._robot2, self.scene.num_envs)
        factory_utils.set_friction(self._robot2, self.cfg_task.robot_cfg.friction, self.scene.num_envs)

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, GROUND_Z))

        desk_cfg = sim_utils.UsdFileCfg(
            usd_path=DESK005_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        )
        desk_cfg.func(
            "/World/envs/env_.*/Table",
            desk_cfg,
            translation=(0.55, 0.0, DESK005_TABLE_BASE_Z),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        robot2_cfg = copy.deepcopy(self.cfg.robot)
        robot2_cfg.prim_path = "/World/envs/env_.*/Robot2"
        robot2_cfg.init_state.pos = (0.98, 0.0, DESK005_TABLETOP_Z)
        robot2_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        for joint_idx, joint_pos in enumerate(ROBOT2_INITIAL_JOINT_POS, start=1):
            robot2_cfg.init_state.joint_pos[f"panda_joint{joint_idx}"] = joint_pos
        robot2_cfg.init_state.joint_pos["panda_finger_joint1"] = 0.04
        robot2_cfg.init_state.joint_pos["panda_finger_joint2"] = 0.04

        self._robot = Articulation(self.cfg.robot)
        self._robot2 = Articulation(robot2_cfg)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["robot2"] = self._robot2
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _init_tensors(self):
        super()._init_tensors()
        self.robot2_left_finger_body_idx = self._robot2.body_names.index("panda_leftfinger")
        self.robot2_right_finger_body_idx = self._robot2.body_names.index("panda_rightfinger")
        self.robot2_fingertip_body_idx = self._robot2.body_names.index("panda_fingertip_centered")
        self.robot2_ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot2.num_joints), device=self.device)

    def _compute_intermediate_values(self, dt):
        super()._compute_intermediate_values(dt)
        self.robot2_fingertip_midpoint_pos = (
            self._robot2.data.body_pos_w[:, self.robot2_fingertip_body_idx] - self.scene.env_origins
        )
        self.robot2_fingertip_midpoint_quat = self._robot2.data.body_quat_w[:, self.robot2_fingertip_body_idx]
        self.robot2_fingertip_midpoint_linvel = self._robot2.data.body_lin_vel_w[:, self.robot2_fingertip_body_idx]
        self.robot2_fingertip_midpoint_angvel = self._robot2.data.body_ang_vel_w[:, self.robot2_fingertip_body_idx]
        self.robot2_joint_pos = self._robot2.data.joint_pos.clone()
        self.robot2_joint_vel = self._robot2.data.joint_vel.clone()
        robot2_jacobians = self._robot2.root_physx_view.get_jacobians()
        left_jac = robot2_jacobians[:, self.robot2_left_finger_body_idx - 1, 0:6, 0:7]
        right_jac = robot2_jacobians[:, self.robot2_right_finger_body_idx - 1, 0:6, 0:7]
        self.robot2_fingertip_midpoint_jacobian = (left_jac + right_jac) * 0.5
        self.robot2_arm_mass_matrix = self._robot2.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]


def register_dual_franka_factory_env(task_id: str = DUAL_TASK_ID):
    try:
        gym.spec(task_id)
        return
    except gym.error.Error:
        pass
    gym.register(
        id=task_id,
        entry_point=DualFrankaFactoryEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": FactoryTaskPegInsertCfg,
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        },
    )


def register_dual_franka_factory_nut_env(task_id: str = DUAL_NUT_THREAD_TASK_ID):
    try:
        gym.spec(task_id)
        return
    except gym.error.Error:
        pass
    gym.register(
        id=task_id,
        entry_point=DualFrankaFactoryEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": FactoryTaskNutThreadCfg,
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        },
    )


def configure_factory_dual_franka_env_cfg(env_cfg, args):
    env_cfg.scene.num_envs = args.num_envs
    if getattr(env_cfg, "task_name", None) == "nut_thread":
        env_cfg.task.held_asset.spawn.rigid_props.disable_gravity = False
    camera_translate = tuple(getattr(args, "wrist_camera_translate", (0.05, 0.0, -0.02)))
    camera_rotate_zyx = tuple(getattr(args, "wrist_camera_rotate_zyx", (0.0, 90.0, 180.0)))
    env_cfg.robot.spawn.usd_path = _ensure_franka_camera_usd(
        env_cfg.robot.spawn.usd_path,
        camera_translate=camera_translate,
        camera_rotate_zyx=camera_rotate_zyx,
    )
    if getattr(args, "disable_fabric", False):
        env_cfg.scene.clone_in_fabric = False
    if getattr(args, "record_camera", "viewer") == "franka1_d455":
        env_cfg.viewer.cam_prim_path = f"/World/envs/env_0/Robot/{D455_COLOR_CAMERA_REL_PATH}"
    elif getattr(args, "record_camera", "viewer") == "franka2_d455":
        env_cfg.viewer.cam_prim_path = f"/World/envs/env_0/Robot2/{D455_COLOR_CAMERA_REL_PATH}"
    env_cfg.robot.init_state.pos = (
        env_cfg.robot.init_state.pos[0] + TASK_X_OFFSET,
        env_cfg.robot.init_state.pos[1],
        env_cfg.robot.init_state.pos[2] + DESK005_TABLETOP_Z,
    )
    env_cfg.task.fixed_asset.init_state.pos = (
        env_cfg.task.fixed_asset.init_state.pos[0] + TASK_X_OFFSET,
        env_cfg.task.fixed_asset.init_state.pos[1],
        env_cfg.task.fixed_asset.init_state.pos[2] + DESK005_TABLETOP_Z,
    )
    env_cfg.task.held_asset.init_state.pos = (
        env_cfg.task.held_asset.init_state.pos[0] + TASK_X_OFFSET,
        env_cfg.task.held_asset.init_state.pos[1],
        env_cfg.task.held_asset.init_state.pos[2] + DESK005_TABLETOP_Z,
    )
