"""
DIAMBRA Hackathon - Agent Inference Script
This script is used for submission to the DIAMBRA competition platform.

Usage:
    diambra run python agent.py
"""

import os
from diambra.arena import Roles, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from stable_baselines3 import PPO


def main():
    """Run the trained agent for evaluation."""
    
    # ==================== ENVIRONMENT SETTINGS ====================
    # These MUST match training settings exactly!
    
    settings = EnvironmentSettings()
    settings.frame_shape = (128, 128, 1)  # Grayscale 128x128
    settings.action_space = SpaceTypes.MULTI_DISCRETE
    settings.step_ratio = 6
    settings.role = Roles.P1  # Play as Player 1
    
    # ==================== WRAPPER SETTINGS ====================
    # These MUST match training settings exactly!
    
    wrappers_settings = WrappersSettings()
    
    # Temporal/Action History
    wrappers_settings.stack_frames = 4
    wrappers_settings.dilation = 1
    wrappers_settings.add_last_action = True
    wrappers_settings.stack_actions = 12
    
    # Normalization
    wrappers_settings.scale = True
    wrappers_settings.exclude_image_scaling = True
    wrappers_settings.normalize_reward = False  # IMPORTANT: False for inference!
    
    # SB3 Compatibility
    wrappers_settings.role_relative = True
    wrappers_settings.flatten = True
    wrappers_settings.filter_keys = [
        "action",
        "own_health",
        "opp_health",
        "own_side",
        "opp_side",
        "opp_character",
        "stage",
        "timer"
    ]
    
    # ==================== CREATE ENVIRONMENT ====================
    
    env, num_envs = make_sb3_env(
        "sfiii3n",  # Street Fighter III: 3rd Strike
        settings,
        wrappers_settings,
        no_vec=True  # Single environment for inference
    )
    print(f"Activated {num_envs} environment(s)")
    
    # ==================== LOAD MODEL ====================
    
    # Model path relative to this script
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "diambra_agent_final"
    )
    
    print(f"Loading model from: {model_path}")
    agent = PPO.load(model_path)
    
    # Print policy architecture
    print("Policy architecture:")
    print(agent.policy)
    
    # ==================== RUN AGENT ====================
    
    print("\nStarting agent execution...")
    
    obs, info = env.reset()
    total_reward = 0
    episode_count = 0
    
    while True:
        # Get action from trained agent
        action, _ = agent.predict(obs, deterministic=False)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action.tolist())
        total_reward += reward
        
        # Check for episode end
        if terminated or truncated:
            episode_count += 1
            print(f"Episode {episode_count} finished. Reward: {total_reward:.2f}")
            total_reward = 0
            
            obs, info = env.reset()
            
            # Check if game is completely done (all rounds/matches)
            if info.get("env_done", False):
                break
    
    print(f"\nAgent execution completed. Total episodes: {episode_count}")
    
    # Close environment
    env.close()
    
    return 0


if __name__ == "__main__":
    main()