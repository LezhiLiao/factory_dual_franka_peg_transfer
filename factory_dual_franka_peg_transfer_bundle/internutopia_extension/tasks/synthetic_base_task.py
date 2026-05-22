"""Reusable helpers for checkpoint-driven synthetic-data tasks.

This base class intentionally keeps task semantics out of the recording path:
subclasses implement ``play_once()`` and ``check_success()``, while this module
owns RL-Games checkpoint loading, policy rollout, RGB capture, HDF5 writing,
and MP4 writing.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np
import torch


DEFAULT_VIEWER_EYE = (-0.1, 1.2, 1.0)
DEFAULT_VIEWER_LOOKAT = (0.3, 0.0, 0.15)


def add_synthetic_data_args(parser):
    parser.add_argument("--video_fps", type=int, default=30, help="MP4 frame rate.")
    parser.add_argument("--video_width", type=int, default=640, help="Viewer render width.")
    parser.add_argument("--video_height", type=int, default=480, help="Viewer render height.")
    parser.add_argument("--video_frame_repeat", type=int, default=3, help="Repeat each captured RGB frame in the MP4.")
    parser.add_argument(
        "--record_camera",
        choices=("viewer", "franka1_wrist_viewer", "franka2_wrist_viewer", "franka1_d455", "franka2_d455"),
        default="viewer",
        help="Camera source used for MP4 frames. Non-viewer modes also save observations/wrist_rgb in HDF5.",
    )
    parser.add_argument("--viewer_eye", type=float, nargs=3, default=DEFAULT_VIEWER_EYE, help="Viewer camera eye.")
    parser.add_argument(
        "--viewer_lookat", type=float, nargs=3, default=DEFAULT_VIEWER_LOOKAT, help="Viewer camera target."
    )


def configure_viewer(env_cfg, args):
    env_cfg.viewer.eye = tuple(args.viewer_eye)
    env_cfg.viewer.lookat = tuple(args.viewer_lookat)
    env_cfg.viewer.resolution = (args.video_width, args.video_height)


def to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_rgb_array(frame) -> np.ndarray:
    rgb = np.asarray(frame)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise RuntimeError(f"Expected an RGB/RGBA frame, got shape={rgb.shape}.")
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb).copy()


@dataclass
class SyntheticRolloutBuffer:
    phase: list[int] = field(default_factory=list)
    action: list[np.ndarray] = field(default_factory=list)
    success: list[np.ndarray] = field(default_factory=list)
    observations: dict[str, list[np.ndarray]] = field(default_factory=dict)
    image_observations: dict[str, list[np.ndarray]] = field(default_factory=dict)
    frames: list[np.ndarray] = field(default_factory=list)

    def append(
        self,
        phase: int,
        action: np.ndarray,
        success: np.ndarray,
        observations: dict[str, np.ndarray],
        image_observations: dict[str, np.ndarray] | None,
        frame: np.ndarray | None,
        frame_repeat: int,
    ):
        self.phase.append(phase)
        self.action.append(action)
        self.success.append(success)
        for name, value in observations.items():
            self.observations.setdefault(name, []).append(value)
        for name, value in (image_observations or {}).items():
            self.image_observations.setdefault(name, []).append(value)
        if frame is not None:
            rgb = to_rgb_array(frame)
            for _ in range(max(1, int(frame_repeat))):
                self.frames.append(rgb.copy())

    def save_hdf5(self, path: str, metadata: dict[str, Any]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as h5:
            h5.attrs["metadata"] = json.dumps(metadata, sort_keys=True)
            h5.attrs["task"] = metadata["task"]
            h5.attrs["checkpoint"] = metadata["checkpoint"]
            h5.attrs["success"] = bool(metadata["success"])
            h5.attrs["phase_names"] = json.dumps(metadata["phase_names"], sort_keys=True)

            h5.create_dataset("phase", data=np.asarray(self.phase, dtype=np.int32), compression="gzip")
            h5.create_dataset(
                "actions/policy_or_scripted", data=np.asarray(self.action, dtype=np.float32), compression="gzip"
            )
            h5.create_dataset("success", data=np.asarray(self.success, dtype=np.float32), compression="gzip")

            obs_group = h5.create_group("observations")
            for name, values in sorted(self.observations.items()):
                obs_group.create_dataset(name, data=np.asarray(values, dtype=np.float32), compression="gzip")
            for name, values in sorted(self.image_observations.items()):
                obs_group.create_dataset(name, data=np.asarray(values, dtype=np.uint8), compression="gzip")

    def save_video(self, path: str, fps: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not self.frames:
            raise RuntimeError("No RGB frames were captured; cannot write MP4.")
        imageio.mimsave(path, self.frames, fps=fps, macro_block_size=1)


class RlGamesCheckpointSkill:
    """Checkpoint-driven RL-Games policy runner for IsaacLab vector envs."""

    def __init__(self, raw_env, env, agent, resume_path: str):
        self.raw_env = raw_env
        self.raw = raw_env.unwrapped
        self.env = env
        self.agent = agent
        self.resume_path = resume_path

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = self.agent.get_batch_size(obs, 1)
        if self.agent.is_rnn:
            self.agent.init_rnn()
        return obs

    def step(self, obs):
        with torch.inference_mode():
            agent_obs = self.agent.obs_to_torch(obs)
            actions = self.agent.get_action(agent_obs, is_deterministic=self.agent.is_deterministic)
            next_obs, rewards, dones, info = self.env.step(actions)
        return next_obs, rewards, dones, info, actions

    def run_until_success(self, task, phase: int, max_steps: int, stop_on_success: bool = True, obs=None):
        current_obs = self.reset() if obs is None else obs
        last_reward = None
        last_done = None
        for _ in range(max(1, int(max_steps))):
            current_obs, rewards, dones, _, actions = self.step(current_obs)
            last_reward = rewards
            last_done = dones
            task.record(phase, actions)
            if stop_on_success and task.check_success(require=False):
                break
        return current_obs, last_reward, last_done


def create_rl_games_checkpoint_skill(task_name: str, env_cfg, agent_cfg: dict, checkpoint: str) -> RlGamesCheckpointSkill:
    """Create an IsaacLab env, wrap it for RL-Games, and restore a policy checkpoint."""

    import gymnasium as gym
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.player import BasePlayer
    from rl_games.torch_runner import Runner

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    resume_path = retrieve_file_path(checkpoint)
    env_cfg.log_dir = os.path.dirname(os.path.dirname(resume_path))

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    raw_env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)

    env = RlGamesVecEnvWrapper(raw_env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    return RlGamesCheckpointSkill(raw_env=raw_env, env=env, agent=agent, resume_path=resume_path)


class SyntheticDataBaseTask:
    """Base class for IsaacLab-backed synthetic-data tasks in InternUtopia."""

    def __init__(
        self,
        rl_skill: RlGamesCheckpointSkill,
        task_name: str,
        output_dir: str,
        video_fps: int,
        video_frame_repeat: int,
        record_camera: str = "viewer",
        viewer_eye=DEFAULT_VIEWER_EYE,
        viewer_lookat=DEFAULT_VIEWER_LOOKAT,
    ):
        self.rl_skill = rl_skill
        self.raw_env = rl_skill.raw_env
        self.raw = rl_skill.raw
        self.env = rl_skill.env
        self.agent = rl_skill.agent
        self.resume_path = rl_skill.resume_path
        self.task_name = task_name
        self.output_dir = output_dir
        self.video_fps = video_fps
        self.video_frame_repeat = video_frame_repeat
        self.record_camera = record_camera
        self.viewer_eye = tuple(viewer_eye)
        self.viewer_lookat = tuple(viewer_lookat)
        self.rollout = SyntheticRolloutBuffer()
        self.info: dict[str, Any] = {}

    def reset_scene(self):
        return self.rl_skill.reset()

    def run_checkpoint_policy(self, phase: int, max_steps: int, stop_on_success: bool = True, obs=None):
        return self.rl_skill.run_until_success(
            task=self,
            phase=phase,
            max_steps=max_steps,
            stop_on_success=stop_on_success,
            obs=obs,
        )

    def record(self, phase: int, action=None):
        if hasattr(self.raw, "_compute_intermediate_values"):
            self.raw._compute_intermediate_values(self.raw.physics_dt)
        self.prepare_render_camera()
        frame = self.render_frame()
        self.rollout.append(
            phase=phase,
            action=self._action_array(action),
            success=self.success_array(),
            observations=self.collect_observations(),
            image_observations=self.collect_image_observations(frame),
            frame=frame,
            frame_repeat=self.video_frame_repeat,
        )

    def prepare_render_camera(self):
        pass

    def collect_observations(self) -> dict[str, np.ndarray]:
        raw = self.raw
        observations = {
            "joint_pos": to_numpy(raw.joint_pos),
            "joint_vel": to_numpy(raw.joint_vel),
            "ee_pos": to_numpy(raw.fingertip_midpoint_pos),
            "ee_quat": to_numpy(raw.fingertip_midpoint_quat),
            "ee_linvel": to_numpy(raw.fingertip_midpoint_linvel),
            "ee_angvel": to_numpy(raw.fingertip_midpoint_angvel),
            "fixed_pos": to_numpy(raw.fixed_pos),
            "fixed_quat": to_numpy(raw.fixed_quat),
            "fixed_tip_pos": to_numpy(raw.fixed_pos_obs_frame),
            "held_pos": to_numpy(raw.held_pos),
            "held_quat": to_numpy(raw.held_quat),
        }
        if hasattr(raw, "robot2_fingertip_midpoint_pos"):
            observations.update(
                {
                    "robot2_joint_pos": to_numpy(raw.robot2_joint_pos),
                    "robot2_joint_vel": to_numpy(raw.robot2_joint_vel),
                    "robot2_ee_pos": to_numpy(raw.robot2_fingertip_midpoint_pos),
                    "robot2_ee_quat": to_numpy(raw.robot2_fingertip_midpoint_quat),
                    "robot2_ee_linvel": to_numpy(raw.robot2_fingertip_midpoint_linvel),
                    "robot2_ee_angvel": to_numpy(raw.robot2_fingertip_midpoint_angvel),
                }
            )
        return observations

    def render_frame(self) -> np.ndarray:
        return to_rgb_array(self.raw_env.render())

    def collect_image_observations(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        if self.record_camera == "viewer":
            return {}
        return {"wrist_rgb": to_rgb_array(frame)}

    def success_array(self) -> np.ndarray:
        return np.asarray([[float(self.check_success(require=False))]], dtype=np.float32)

    def check_success(self, require: bool = True) -> bool:
        raise NotImplementedError

    def save_outputs(self, success: bool, stem_prefix: str, phase_names: dict[str, str], metadata: dict[str, Any]):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        stem = f"{stem_prefix}_{timestamp}_{'success' if success else 'fail'}"
        hdf5_path = os.path.join(self.output_dir, f"{stem}.hdf5")
        video_path = os.path.join(self.output_dir, f"{stem}.mp4")
        full_metadata = {
            "task": self.task_name,
            "checkpoint": self.resume_path,
            "runtime": "InternUtopia .venv with IsaacLab task launched from /data/user/InternUtopia",
            "viewer_eye": list(self.viewer_eye),
            "viewer_lookat": list(self.viewer_lookat),
            "video_fps": self.video_fps,
            "video_frame_repeat": self.video_frame_repeat,
            "record_camera": self.record_camera,
            "phase_names": phase_names,
            "success": success,
        }
        full_metadata.update(metadata)
        self.rollout.save_hdf5(hdf5_path, full_metadata)
        self.rollout.save_video(video_path, self.video_fps)
        return hdf5_path, video_path

    def _action_array(self, action) -> np.ndarray:
        if action is not None:
            return to_numpy(action)
        action_dim = getattr(getattr(self.raw, "cfg", None), "action_space", 6)
        return np.zeros((self.raw.num_envs, int(action_dim)), dtype=np.float32)
