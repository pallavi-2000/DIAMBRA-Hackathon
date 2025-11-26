# 🎮 Street Fighter III AI Agent - DIAMBRA Hackathon

An AI agent trained to play Street Fighter III: 3rd Strike using Deep Reinforcement Learning (PPO algorithm) for the DIAMBRA/0g Cambridge AI Hackathon.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🏆 Hackathon

This project was developed for the **DIAMBRA/0g Cambridge University AI Hackathon** (November 2025), where participants train RL agents to compete in Street Fighter III battles.

## 🚀 Features

- **PPO Algorithm** with optimized hyperparameters for fighting games
- **MultiInputPolicy** - Uses both visual frames AND RAM states (health, position, etc.)
- **Frame Stacking** (4 frames) - Motion detection and temporal awareness
- **Action History** (12 actions) - Enables combo learning
- **Linear Learning Rate Decay** - Better convergence (2.5e-4 → 2.5e-6)
- **Linear Clip Range Decay** - Adaptive policy updates (0.15 → 0.025)
- **Reward Normalization** - Stable training signal
- **Engine-level Frame Processing** - Grayscale 128x128 for speed
- **Parallel Environment Training** - Up to 8x faster with multiple CPU cores

## 📊 Training Metrics

| Metric | Value |
|--------|-------|
| Algorithm | PPO |
| Policy | MultiInputPolicy |
| Total Timesteps | 1,000,000 |
| Frame Shape | 128x128 Grayscale |
| Frame Stack | 4 |
| Action History | 12 |
| Gamma | 0.94 |
| Learning Rate | 2.5e-4 → 2.5e-6 |
| Batch Size | 256 |

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Docker (for DIAMBRA Arena)
- Street Fighter III ROM file

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install diambra-arena[stable-baselines3]
pip install torch tensorboard

# Set ROM path
export DIAMBRAROMSPATH=/path/to/your/roms/
```

## 🎯 Usage

### Training

```bash
# Train with 6 parallel environments (adjust based on CPU cores)
diambra run -s=6 python train_agent_v2_fast.py

# Monitor training with TensorBoard
tensorboard --logdir=./logs_agent2
```

### Inference / Testing

```bash
# Run trained agent
diambra run python agent.py
```

## 📁 Project Structure

```
├── train_agent_v2_fast.py   # Main training script (CPU optimized)
├── agent.py                  # Inference script for submission
├── models_agent2/            # Saved models
│   ├── agent2_final.zip
│   └── config.json
├── checkpoints_agent2/       # Training checkpoints
├── logs_agent2/              # TensorBoard logs
└── README.md
```

## ⚙️ Configuration

Key hyperparameters in `train_agent_v2_fast.py`:

```python
# Environment
frame_shape = (128, 128, 1)      # Grayscale
stack_frames = 4                  # Temporal awareness
stack_actions = 12                # Action history

# PPO Hyperparameters
gamma = 0.94                      # Discount factor (fighting games)
learning_rate = 2.5e-4 → 2.5e-6  # Linear decay
clip_range = 0.15 → 0.025        # Linear decay
batch_size = 256
n_steps = 128
n_epochs = 4

# RAM State Features
filter_keys = [
    "own_health", "opp_health",
    "own_side", "opp_side",
    "opp_character", "stage", "timer"
]
```

## 📈 Results

Training progress monitored via TensorBoard:

- **ep_rew_mean**: Average episode reward (increasing = learning)
- **explained_variance**: Value function accuracy (higher = better)
- **fps**: Training speed
- **loss**: Overall training loss

## 🔗 Resources

- [DIAMBRA Arena Documentation](https://docs.diambra.ai/)
- [DIAMBRA GitHub](https://github.com/diambra/arena)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [0G Labs](https://0g.ai/)

## 📝 Submission

For hackathon submission:

1. Train agent to completion
2. Upload model to 0G Storage:
   ```bash
   0g-storage upload models_agent2/
   ```
3. Save CID for submission form

## 👤 Author

**Your Name**
- Cambridge University AI Hackathon Participant
- [GitHub](https://github.com/pallavi-2000)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- DIAMBRA Team for the amazing RL platform
- 0G Labs for storage solutions
- Mistral for LLM support
- Cambridge University for hosting

---

*Built with ❤️ for the DIAMBRA/0g Cambridge AI Hackathon 2025*
