import diambra.arena
import gymnasium as gym
from gymnasium import ObservationWrapper, ActionWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import numpy as np
import os
from pathlib import Path


# ---- Custom Wrappers ----
class ScreenOnlyWrapper(ObservationWrapper):
    """Extract only the screen frame from DIAMBRA's dict observation."""
    def __init__(self, env):
        super().__init__(env)
        # Print available keys to debug
        print(f"Available observation keys: {env.observation_space.spaces.keys()}")
        
        # DIAMBRA returns frame data - find the correct key
        if "frame" in env.observation_space.spaces:
            self.observation_space = env.observation_space["frame"]
            self.frame_key = "frame"
        elif "screen" in env.observation_space.spaces:
            self.observation_space = env.observation_space["screen"]
            self.frame_key = "screen"
        else:
            # Print all available keys
            keys = list(env.observation_space.spaces.keys())
            raise KeyError(f"Could not find frame/screen key. Available keys: {keys}")

    def observation(self, obs):
        return obs[self.frame_key]


class ActionSimplificationWrapper(ActionWrapper):
    """Map discrete actions to DIAMBRA's move/attack format (integers)."""
    def __init__(self, env):
        super().__init__(env)
        # DIAMBRA expects: [move, attack] where both are integers
        # move: 0=idle, 1=forward, 2=backward, 3=crouch
        # attack: 0=none, 1=light punch, 2=medium punch, 3=heavy punch, 
        #         4=light kick, 5=medium kick, 6=heavy kick
        self.moves = {
            0: [0, 0],  # No action
            1: [1, 0],  # Move forward
            2: [2, 0],  # Move backward
            3: [3, 0],  # Crouch
            4: [0, 1],  # Light punch
            5: [0, 3],  # Heavy punch
            6: [0, 4],  # Light kick
            7: [1, 1],  # Forward + Light punch
            8: [1, 3],  # Forward + Heavy punch
            9: [1, 4],  # Forward + Light kick
            10: [2, 1], # Backward + Light punch
            11: [3, 1], # Crouch + Light punch
            12: [3, 3], # Crouch + Heavy punch
        }
        self.action_space = gym.spaces.Discrete(len(self.moves))

    def action(self, act):
        # Convert to Python int to avoid numpy float32 issues
        move, attack = self.moves[int(act)]
        return [int(move), int(attack)]


class RewardShapingWrapper(gym.Wrapper):
    """Shape rewards to encourage aggressive and successful play."""
    def __init__(self, env):
        super().__init__(env)
        self.last_player_health = 100.0
        self.last_opponent_health = 100.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract health values if available in info
        player_health = float(info.get("agent_health", 100.0))
        opponent_health = float(info.get("opponent_health", 100.0))
        
        # Convert reward to float if needed
        reward = float(reward)
        
        # Reward for damaging opponent
        damage_dealt = self.last_opponent_health - opponent_health
        reward += damage_dealt * 0.5
        
        # Penalty for taking damage
        damage_taken = self.last_player_health - player_health
        reward -= damage_taken * 0.3
        
        # Bonus for winning
        if terminated and player_health > 0:
            reward += 100.0
        
        # Penalty for losing
        if terminated and player_health <= 0:
            reward -= 50.0
        
        self.last_player_health = player_health
        self.last_opponent_health = opponent_health
        
        return obs, float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_player_health = 100.0
        self.last_opponent_health = 100.0
        return obs, info


# ---- Environment Factory ----
def make_env():
    """Create a single training environment with all wrappers."""
    def _init():
        env = diambra.arena.make(
            "sfiii3n",
            render_mode="rgb_array"
        )
        env = ScreenOnlyWrapper(env)
        env = ActionSimplificationWrapper(env)
        env = RewardShapingWrapper(env)
        return env
    return _init


# ---- Training Configuration ----
class TrainingConfig:
    """Centralized hyperparameter configuration."""
    # PPO hyperparameters
    learning_rate = 3e-4
    n_steps = 2048
    batch_size = 256
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.1
    ent_coef = 0.01
    vf_coef = 0.5
    
    # Training parameters
    total_timesteps = 1_000_000  # Increased from 500k
    num_envs = 1
    
    # Checkpoint settings
    checkpoint_freq = 50_000  # Save every 50k steps
    eval_freq = 100_000  # Evaluate every 100k steps
    
    # Paths
    model_dir = Path("./models")
    checkpoint_dir = Path("./checkpoints")
    logs_dir = Path("./logs")


def setup_directories():
    """Create necessary directories for checkpoints and logs."""
    for directory in [TrainingConfig.model_dir, 
                      TrainingConfig.checkpoint_dir, 
                      TrainingConfig.logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)


def create_callbacks():
    """Create training callbacks for checkpointing and evaluation."""
    checkpoint_callback = CheckpointCallback(
        save_freq=TrainingConfig.checkpoint_freq,
        save_path=str(TrainingConfig.checkpoint_dir),
        name_prefix="ppo_sfiii3",
        save_replay_buffer=False
    )
    
    return [checkpoint_callback]


def train_agent():
    """Main training loop."""
    print("🎮 Initializing Street Fighter III Training Environment...")
    setup_directories()
    
    # Create vectorized environment
    vec_env = DummyVecEnv([make_env() for _ in range(TrainingConfig.num_envs)])
    
    print("🤖 Creating PPO Agent...")
    # Use a simple run name for TensorBoard
    run_name = f"ppo_sfiii3_{int(np.random.random() * 1e6)}"
    
    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        learning_rate=TrainingConfig.learning_rate,
        n_steps=TrainingConfig.n_steps,
        batch_size=TrainingConfig.batch_size,
        gamma=TrainingConfig.gamma,
        gae_lambda=TrainingConfig.gae_lambda,
        clip_range=TrainingConfig.clip_range,
        ent_coef=TrainingConfig.ent_coef,
        vf_coef=TrainingConfig.vf_coef,
        tensorboard_log=str(TrainingConfig.logs_dir),
        device="cpu",
    )
    
    print("🏋️ Starting Training...")
    print(f"Total timesteps: {TrainingConfig.total_timesteps:,}")
    print(f"Checkpoint frequency: {TrainingConfig.checkpoint_freq:,} steps")
    
    callbacks = create_callbacks()
    
    try:
        model.learn(
            total_timesteps=TrainingConfig.total_timesteps,
            callback=callbacks,
            log_interval=1  # Print logs every iteration for visibility
        )
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    
    # Save final model
    final_model_path = TrainingConfig.model_dir / "ppo_sfiii3_final"
    model.save(str(final_model_path))
    print(f"✅ Final model saved to {final_model_path}")
    
    # Also save in ONNX format for compatibility
    try:
        onnx_path = TrainingConfig.model_dir / "ppo_sfiii3_final.onnx"
        model.save(str(onnx_path), save_format="onnx")
        print(f"✅ ONNX model saved to {onnx_path}")
    except Exception as e:
        print(f"⚠️ Could not save ONNX model: {e}")
    
    vec_env.close()
    print("🔥 Training complete!")


def evaluate_agent(model_path, num_episodes=5):
    """Evaluate a trained agent."""
    print(f"\n📊 Evaluating agent: {model_path}")
    
    env = make_env()()
    model = PPO.load(model_path)
    
    total_reward = 0
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        terminated, truncated = False, False
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    avg_reward = total_reward / num_episodes
    print(f"Average Reward: {avg_reward:.2f}")
    env.close()
    
    return avg_reward


if __name__ == "__main__":
    train_agent()
    
    # Optional: Evaluate the trained agent
    # latest_model = max(TrainingConfig.checkpoint_dir.glob("*.zip"), key=os.path.getctime)
    # evaluate_agent(str(latest_model), num_episodes=3)