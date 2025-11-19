# Copyright (c) 2022-2025, The Isaac Lab Project Developers.  
# All rights reserved.  
# SPDX-License-Identifier: BSD-3-Clause  
  
"""Script to replay demonstrations and record video."""  
  
import argparse  
import sys
from isaaclab.app import AppLauncher  
import cli_args
  
# Parse arguments  
parser = argparse.ArgumentParser(description="Replay demonstrations with video recording.")  
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-v0", help="Task name.")  
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--dataset_file", type=str, default="datasets/dataset.hdf5", help="Path to HDF5 dataset.")  
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")  
parser.add_argument("--video_length", type=int, default=200, help="Maximum length of recorded videos.")  
  
cli_args.add_rsl_rl_args(parser)

# Add AppLauncher arguments (includes --headless, --device, etc.)  
AppLauncher.add_app_launcher_args(parser)  
args_cli, hydra_args = parser.parse_known_args()  

# always enable cameras to record video
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
  
# Launch simulator  
app_launcher = AppLauncher(args_cli)  
simulation_app = app_launcher.app  
  
"""Rest of imports after AppLauncher."""  
  
import contextlib  
import gymnasium as gym  
import os  
import torch  
from datetime import datetime
  
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab.utils.datasets import HDF5DatasetFileHandler  
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):  
    """Replay episodes loaded from a file with video recording."""  

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Load dataset  
    if not os.path.exists(args_cli.dataset_file):  
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")  
      
    dataset_file_handler = HDF5DatasetFileHandler()  
    dataset_file_handler.open(args_cli.dataset_file)  
    episode_count = dataset_file_handler.get_num_episodes()  
      
    if episode_count == 0:  
        print("No episodes found in the dataset.")  
        return  
      
    print(f"Found {episode_count} episodes in dataset")  

    log_root_path = os.path.join("logs", "rsl_rl", "replay_with_video")
    log_root_path = os.path.abspath(log_root_path) 
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  
    log_dir = os.path.join(log_root_path, log_dir)

    env_cfg.log_dir = log_dir

    # Create environment with rgb_array render mode for video recording  
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")  

    # Wrap for video recording  
    video_kwargs = {
        "video_folder": log_dir,
        "step_trigger": lambda step: step % 200 == 0,  # Record every 200 steps
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    print(f"[INFO] Recording videos to: {log_dir}")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
      
    # Get episode names  
    episode_names = list(dataset_file_handler.get_episode_names())  
      
    # Replay each episode  
    with torch.inference_mode():  
        for episode_idx in range(episode_count):  
            print(f"Replaying episode {episode_idx + 1}/{episode_count}")  
              
            # Load episode data  
            episode_data = dataset_file_handler.load_episode(episode_names[episode_idx], env.device)  
              
            # Reset environment to initial state  
            initial_state = episode_data.get_initial_state()  
            env.reset()  
            env.unwrapped.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=True)  
              
            # Replay actions  
            while True:  
                action = episode_data.get_next_action()

                if action is None:  
                    break  
                print("Action shape:", action.shape)

                # Step environment
                obs, rewards, dones, extras = env.step(action.unsqueeze(0))
                print("Observations shape:", obs.shape)
                if dones.any():  # Check if any environment is done
                    break

            print(f"Completed episode {episode_idx + 1}")
      
    # Close environment  
    env.close()  
    dataset_file_handler.close()  
    simulation_app.close()  
    print("Replay complete!")  
  
if __name__ == "__main__":  
    main()