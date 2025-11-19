# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# def object_is_lifted(
#     env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("cube_3")
# ) -> torch.Tensor:
#     """Reward the agent for lifting the object above the minimal height."""
#     object: RigidObject = env.scene[object_cfg.name]
#     return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


# def object_ee_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
# ) -> torch.Tensor:
#     """Reward the agent for reaching the object using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     object: RigidObject = env.scene[object_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     # Target object position: (num_envs, 3)
#     cube_pos_w = object.data.root_pos_w
#     # End-effector position: (num_envs, 3)
#     ee_w = ee_frame.data.target_pos_w[..., 0, :]
#     # Distance of the end-effector to the object: (num_envs,)
#     object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

#     return 1 - torch.tanh(object_ee_distance / std)


# def object_goal_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     minimal_height: float,
#     command_name: str,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
# ) -> torch.Tensor:
#     """Reward the agent for tracking the goal pose using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # compute the desired position in the world frame
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
#     # distance of the end-effector to the object: (num_envs,)
#     distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
#     # rewarded if the object is lifted above the threshold
#     return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel.

    The command provided by the command manager is expected to be expressed
    in the robot base frame (matching mdp.UniformPoseCommandCfg semantics).
    We transform the commanded position into world frame using the robot's
    root pose and compute the distance to the object's world-space root
    position. The reward is active only if the object is above minimal_height.
    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # command shape: (num_envs, command_dim) where first 3 are pos in robot base
    des_pos_b = command[:, :3]
    # transform desired position from robot base to world frame
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance between desired world pos and object world pos
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height).float() * (1 - torch.tanh(distance / std))

def cubes_are_stacked(
    env: ManagerBasedRLEnv, minimal_height: float, top_cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"), bottom_cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1")
) -> torch.Tensor:
    """Reward the agent for stacking the top cube on the bottom cube above a minimal height."""
    top_cube: RigidObject = env.scene[top_cube_cfg.name]
    bottom_cube: RigidObject = env.scene[bottom_cube_cfg.name]
    # Check if the top cube is above the bottom cube and both are above minimal height
    stacked = (top_cube.data.root_pos_w[:, 2] > minimal_height) & (
        torch.abs(top_cube.data.root_pos_w[:, 0] - bottom_cube.data.root_pos_w[:, 0]) < 0.05
    ) & (
        torch.abs(top_cube.data.root_pos_w[:, 1] - bottom_cube.data.root_pos_w[:, 1]) < 0.05
    )
    return stacked.float()


def top_cube_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    top_cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the top cube using tanh-kernel."""
    top_cube: RigidObject = env.scene[top_cube_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = top_cube.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(distance / std)


# def top_cube_goal_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     minimal_height: float,
#     command_name: str,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     top_cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
# ) -> torch.Tensor:
#     """Reward the agent for moving the top cube to the goal pose using tanh-kernel, only if stacked."""
#     robot: RigidObject = env.scene[robot_cfg.name]
#     top_cube: RigidObject = env.scene[top_cube_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
#     distance = torch.norm(des_pos_w - top_cube.data.root_pos_w, dim=1)
#     # Reward only if the top cube is stacked
#     return (top_cube.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
