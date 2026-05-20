"""Dual-Franka IsaacLab Factory scene used by synthetic peg-transfer tasks."""

from __future__ import annotations

import copy
import os

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab_tasks.direct.factory import agents
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DESK005_USD = os.path.join(PROJECT_ROOT, "asset", "desk005", "model_table_7.usd")
GROUND_Z = -1.05
DESK005_TABLE_HEIGHT = 0.744
DESK005_TABLE_BASE_Z = GROUND_Z
DESK005_TABLETOP_Z = DESK005_TABLE_BASE_Z + DESK005_TABLE_HEIGHT
TASK_X_OFFSET = -0.2
DUAL_TASK_ID = "InternUtopia-Factory-DualFrankaPegTransfer-Direct-v0"
ROBOT2_INITIAL_JOINT_POS = (0.0015178, -0.19651, -0.0014364, -1.9761, -0.00027717, 1.7796, 0.78556)


class DualFrankaFactoryEnv(FactoryEnv):
    """Factory peg-in-hole env with a second scripted Franka across desk005."""

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
        self.robot2_joint_pos = self._robot2.data.joint_pos.clone()
        self.robot2_joint_vel = self._robot2.data.joint_vel.clone()
        robot2_jacobians = self._robot2.root_physx_view.get_jacobians()
        left_jac = robot2_jacobians[:, self.robot2_left_finger_body_idx - 1, 0:6, 0:7]
        right_jac = robot2_jacobians[:, self.robot2_right_finger_body_idx - 1, 0:6, 0:7]
        self.robot2_fingertip_midpoint_jacobian = (left_jac + right_jac) * 0.5


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


def configure_factory_dual_franka_env_cfg(env_cfg, args):
    env_cfg.scene.num_envs = args.num_envs
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
