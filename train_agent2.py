"""
DIAMBRA Hackathon - Agent V2 Training Script (CPU OPTIMIZED)
Optimized for Street Fighter III using DIAMBRA best practices.
Configured for maximum CPU core utilization.

Usage:
    diambra run -s=8 python train_agent_v2_fast.py

    -s=8 means 8 parallel environments (adjust based on your CPU cores)
    Recommended: Use (CPU cores - 2) for -s value
    
    Examples:
      8-core CPU:  diambra run -s=6 python train_agent_v2_fast.py
      12-core CPU: diambra run -s=10 python train_agent_v2_fast.py
      16-core CPU: diambra run -s=14 python train_agent_v2_fast.py
"""

import diambra.arena
from diambra.arena import EnvironmentSettings, WrappersSettings, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from pathlib import Path
import numpy as np
import json
import torch
import os


# ==================== CPU OPTIMIZATION ====================

def setup_cpu_optimization():
    """Configure PyTorch for optimal CPU performance."""
    
    # Get number of CPU cores
    num_cores = os.cpu_count()
    print(f"🖥️  Detected {num_cores} CPU cores")
    
    # Set PyTorch to use all available cores for intra-op parallelism
    # Leave some cores for the environments
    torch_threads = max(4, num_cores // 2)
    torch.set_num_threads(torch_threads)
    
    # Set inter-op parallelism (for operations that can run in parallel)
    torch.set_num_interop_threads(2)
    
    # Disable GPU (force CPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    print(f"⚡ PyTorch using {torch_threads} threads for computation")
    print(f"⚡ Inter-op threads: 2")
    
    return num_cores


# ==================== CONFIGURATION ====================

class AgentConfig:
    """Complete configuration using DIAMBRA best practices - CPU OPTIMIZED."""
    
    # ENVIRONMENT SETTINGS
    game_id = "sfiii3n"
    frame_shape = (128, 128, 1)  # Grayscale 128x128 - ENGINE LEVEL RESIZE!
    action_space = SpaceTypes.MULTI_DISCRETE
    step_ratio = 6
    difficulty = None  # Random difficulty for robustness
    
    # PARALLEL TRAINING - KEY FOR SPEED!
    # This will be auto-detected, but DIAMBRA CLI -s flag overrides it
    # The actual number comes from make_sb3_env detecting DIAMBRA_ENVS
    num_envs = 8  # Target: 8 parallel environments (adjust with -s flag)
    
    # WRAPPER SETTINGS (CRITICAL!)
    stack_frames = 4              # Temporal info (motion awareness)
    dilation = 1                  # Every frame
    add_last_action = True        # Action context
    stack_actions = 12            # Action history (learn combos!)
    scale = True                  # Normalize observations
    exclude_image_scaling = True  # SB3 handles image normalization
    normalize_reward = True       # Stabilize reward scale
    normalization_factor = 0.5   # DIAMBRA recommended
    role_relative = True          # own/opp instead of P1/P2
    flatten = True                # Required for SB3!
    
    # RAM STATE FILTERING (give agent explicit info!)
    filter_keys = [
        "action",          # What action we just took
        "own_health",      # Our health
        "opp_health",      # Opponent health  
        "own_side",        # Our position
        "opp_side",        # Opponent position
        "opp_character",   # Who we're fighting
        "stage",           # Which stage
        "timer"            # Time remaining
    ]
    
    # PPO HYPERPARAMETERS (DIAMBRA recommended values)
    learning_rate_start = 2.5e-4
    learning_rate_end = 2.5e-6   # Linear decay
    clip_range_start = 0.15
    clip_range_end = 0.025       # Linear decay
    gamma = 0.94                 # Lower for fighting games!
    batch_size = 256             # DIAMBRA recommended
    n_steps = 128                # Smaller rollouts for faster updates
    n_epochs = 4                 # Number of training epochs per rollout
    ent_coef = 0.01              # Exploration bonus
    vf_coef = 0.5                # Value function weight
    
    # TRAINING PARAMETERS
    total_timesteps = 1_000_000  # Target timesteps
    checkpoint_freq = 250_000    # Save every 250k steps
    
    # DIRECTORIES
    model_dir = Path("./models_agent2")
    checkpoint_dir = Path("./checkpoints_agent2")
    logs_dir = Path("./logs_agent2")
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary for saving."""
        return {
            "game_id": cls.game_id,
            "frame_shape": cls.frame_shape,
            "action_space": "MULTI_DISCRETE",
            "step_ratio": cls.step_ratio,
            "difficulty": cls.difficulty,
            "num_envs": cls.num_envs,
            "stack_frames": cls.stack_frames,
            "dilation": cls.dilation,
            "add_last_action": cls.add_last_action,
            "stack_actions": cls.stack_actions,
            "scale": cls.scale,
            "exclude_image_scaling": cls.exclude_image_scaling,
            "normalize_reward": cls.normalize_reward,
            "normalization_factor": cls.normalization_factor,
            "role_relative": cls.role_relative,
            "flatten": cls.flatten,
            "filter_keys": cls.filter_keys,
            "learning_rate_start": cls.learning_rate_start,
            "learning_rate_end": cls.learning_rate_end,
            "clip_range_start": cls.clip_range_start,
            "clip_range_end": cls.clip_range_end,
            "gamma": cls.gamma,
            "batch_size": cls.batch_size,
            "n_steps": cls.n_steps,
            "n_epochs": cls.n_epochs,
            "ent_coef": cls.ent_coef,
            "vf_coef": cls.vf_coef,
            "total_timesteps": cls.total_timesteps,
            "checkpoint_freq": cls.checkpoint_freq,
        }


# ==================== CALLBACKS ====================

class FPSMonitorCallback(BaseCallback):
    """Callback to monitor and display FPS during training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        return True


def linear_schedule(initial_value: float, final_value: float):
    """Linear learning rate/clip range schedule."""
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func


def setup_directories():
    """Create necessary directories."""
    for directory in [AgentConfig.model_dir, 
                      AgentConfig.checkpoint_dir, 
                      AgentConfig.logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)


def save_config(num_envs_actual):
    """Save configuration to JSON file for reproducibility."""
    config = AgentConfig.to_dict()
    config["num_envs_actual"] = num_envs_actual  # Record actual envs used
    config_path = AgentConfig.model_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"📄 Configuration saved to {config_path}")


# ==================== ENVIRONMENT SETUP ====================

def create_environment():
    """Create environment using DIAMBRA best practices."""
    
    # Environment Settings
    settings = EnvironmentSettings()
    settings.frame_shape = AgentConfig.frame_shape  # ENGINE-level resize!
    settings.action_space = AgentConfig.action_space
    settings.step_ratio = AgentConfig.step_ratio
    settings.difficulty = AgentConfig.difficulty
    
    # Wrapper Settings
    wrappers_settings = WrappersSettings()
    
    # Temporal/Action History
    wrappers_settings.stack_frames = AgentConfig.stack_frames
    wrappers_settings.dilation = AgentConfig.dilation
    wrappers_settings.add_last_action = AgentConfig.add_last_action
    wrappers_settings.stack_actions = AgentConfig.stack_actions
    
    # Normalization
    wrappers_settings.scale = AgentConfig.scale
    wrappers_settings.exclude_image_scaling = AgentConfig.exclude_image_scaling
    wrappers_settings.normalize_reward = AgentConfig.normalize_reward
    wrappers_settings.normalization_factor = AgentConfig.normalization_factor
    
    # SB3 Compatibility
    wrappers_settings.role_relative = AgentConfig.role_relative
    wrappers_settings.flatten = AgentConfig.flatten
    wrappers_settings.filter_keys = AgentConfig.filter_keys
    
    # Create environment - DIAMBRA will auto-detect parallel envs from CLI
    # use_subprocess=False for Mac compatibility (DummyVecEnv)
    # use_subprocess=True for Linux (SubprocVecEnv - faster)
    env, num_envs = make_sb3_env(
        AgentConfig.game_id,
        settings,
        wrappers_settings,
        start_index=0,
        allow_early_resets=True,
        use_subprocess=False,  # DummyVecEnv for Mac compatibility
    )
    
    return env, num_envs


# ==================== TRAINING ====================

def train_agent():
    """Train agent using DIAMBRA best practices with CPU optimization."""
    
    print("=" * 70)
    print("🚀 AGENT V2 - CPU OPTIMIZED TRAINING")
    print("=" * 70)
    
    # Setup CPU optimization
    num_cores = setup_cpu_optimization()
    
    setup_directories()
    
    # Create environment
    print("\n🎮 Creating environment...")
    env, num_envs = create_environment()
    print(f"✅ Created {num_envs} parallel environments")
    
    # Adjust batch size based on number of environments
    effective_batch_size = min(AgentConfig.batch_size, AgentConfig.n_steps * num_envs)
    
    # Save config with actual env count
    save_config(num_envs)
    
    print("\n📋 Configuration:")
    print(f"  Game: {AgentConfig.game_id}")
    print(f"  Frame shape: {AgentConfig.frame_shape} (grayscale)")
    print(f"  Parallel envs: {num_envs}")
    print(f"  Frame stacking: {AgentConfig.stack_frames}")
    print(f"  Action history: {AgentConfig.stack_actions}")
    print(f"  Batch size: {effective_batch_size}")
    print(f"  n_steps: {AgentConfig.n_steps}")
    print(f"  Buffer size per update: {AgentConfig.n_steps * num_envs}")
    
    # Create PPO agent
    print("\n🤖 Creating PPO agent...")
    
    lr_schedule = linear_schedule(
        AgentConfig.learning_rate_start,
        AgentConfig.learning_rate_end
    )
    
    clip_schedule = linear_schedule(
        AgentConfig.clip_range_start,
        AgentConfig.clip_range_end
    )
    
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=lr_schedule,
        clip_range=clip_schedule,
        gamma=AgentConfig.gamma,
        batch_size=effective_batch_size,
        n_steps=AgentConfig.n_steps,
        n_epochs=AgentConfig.n_epochs,
        ent_coef=AgentConfig.ent_coef,
        vf_coef=AgentConfig.vf_coef,
        verbose=1,
        tensorboard_log=str(AgentConfig.logs_dir),
        device="cpu",  # Force CPU
    )
    
    # Estimate training time
    # With more envs, FPS scales roughly linearly
    expected_fps = 8 * num_envs  
    expected_hours = AgentConfig.total_timesteps / (expected_fps * 3600)
    
    print("\n⏱️  Training estimates:")
    print(f"  Total timesteps: {AgentConfig.total_timesteps:,}")
    print(f"  Parallel environments: {num_envs}")
    print(f"  Expected FPS: ~{expected_fps} (8 per env × {num_envs} envs)")
    print(f"  Estimated time: ~{expected_hours:.1f} hours")
    print(f"  Comparison: Agent 1 took 31 hours at 8 FPS")
    
    if num_envs >= 4:
        speedup = num_envs / 1  # Compared to single env
        print(f"  🚀 Expected speedup: ~{speedup}x faster than Agent 1!")
    
    print("\n" + "=" * 70)
    print("🏋️  STARTING TRAINING...")
    print("=" * 70 + "\n")
    
    # Checkpoint callback - adjust frequency for parallel envs
    checkpoint_callback = CheckpointCallback(
        save_freq=max(AgentConfig.checkpoint_freq // num_envs, 1000),
        save_path=str(AgentConfig.checkpoint_dir),
        name_prefix="agent2",
        save_replay_buffer=False
    )
    
    # Train!
    try:
        model.learn(
            total_timesteps=AgentConfig.total_timesteps,
            callback=checkpoint_callback,
            log_interval=10,
            progress_bar=True,  # Show progress bar
        )
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        interrupted_path = AgentConfig.model_dir / "agent2_interrupted"
        model.save(str(interrupted_path))
        print(f"💾 Interrupted model saved to {interrupted_path}")
    
    # Save final model
    final_path = AgentConfig.model_dir / "agent2_final"
    model.save(str(final_path))
    print(f"\n✅ Final model saved to {final_path}")
    
    env.close()
    
    print("\n" + "=" * 70)
    print("🔥 TRAINING COMPLETE!")
    print("=" * 70)
    
    print("\n📊 Agent 2 Improvements over Agent 1:")
    print("  ✅ MultiInputPolicy (uses RAM states)")
    print("  ✅ Engine-level frame resizing")
    print("  ✅ Linear LR decay (2.5e-4 → 2.5e-6)")
    print("  ✅ Linear clip decay (0.15 → 0.025)")
    print("  ✅ Fighting-game gamma (0.94)")
    print("  ✅ Parallel environments for speed")
    
    print("\n📦 Files saved:")
    print(f"  Model: {final_path}.zip")
    print(f"  Config: {AgentConfig.model_dir}/config.json")
    print(f"  Checkpoints: {AgentConfig.checkpoint_dir}/")
    print(f"  TensorBoard: {AgentConfig.logs_dir}/")


# ==================== MAIN ====================

if __name__ == "__main__":
    train_agent()