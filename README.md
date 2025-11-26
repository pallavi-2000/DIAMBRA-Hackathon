# 🎮 Street Fighter III AI Agent – DIAMBRA / 0G Cambridge Hackathon

An end-to-end Deep Reinforcement Learning project where I trained an AI agent to play **Street Fighter III: 3rd Strike** using **DIAMBRA Arena + PPO**, developed for the **DIAMBRA / 0G Cambridge University AI Hackathon (Nov 2025).**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Status](https://img.shields.io/badge/Agent%202-Training%20In%20Progress-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

# 📖 Project Overview

This repository documents my **complete learning journey** from building **Agent 1** (slow, inefficient, poorly structured) to **Agent 2**, which adopts best practices from DIAMBRA documentation and RL research.

The goal:
👉 Train a competitive agent that can fight in Street Fighter III using optimized PPO, parallel environments, RAM state features, temporal memory, and better hyperparameters.

---

                         ┌──────────────────────────────────────┐
                         │          Street Fighter III           │
                         │    (DIAMBRA Engine - C++ backend)     │
                         └──────────────────────────────────────┘
                                        ▲
                                        │ Frames (128×128×1)
                                        │ RAM States (health, timer, side,…)
                                        ▼

┌────────────────────────────────────────────────────────────────────────────┐
│                      DIAMBRA Arena + WrappersSettings                      │
│                                                                            │
│  • Engine-level grayscale frame resize (128×128×1)                         │
│  • Frame stacking (4)                                                      │
│  • Action history stacking (12)                                            │
│  • Reward normalization                                                     │
│  • RAM state extraction (8 keys)                                           │
│  • Filtering → {'frame_stack', 'ram_state', 'last_actions'}                │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼

┌────────────────────────────────────────────────────────────────────────────┐
│                   Stable-Baselines3 Vector Environments                    │
│                                                                            │
│  diambra run -s=6                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  │ Env #1        │   │ Env #2        │   │ Env #3        │   │ Env #6        │
│  │ (Docker)      │   │ (Docker)      │   │ (Docker)      │   │ (Docker)      │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
│                                                                            │
│  • Parallel rollout generation (6 envs)                                    │
│  • ~15 FPS combined                                                         │
│  • Faster, more diverse experience                                          │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼

┌────────────────────────────────────────────────────────────────────────────┐
│                  MultiInputPolicy Neural Network (SB3 PPO)                 │
│                                                                            │
│  Inputs:                                                                   │
│   • Frame stack (4 × 128 × 128 grayscale)                                  │
│   • RAM features (health, positions, timer, character, stage…)             │
│   • Action history (12 last actions)                                       │
│                                                                            │
│  Architecture:                                                             │
│   • CNN encoder for vision                                                 │
│   • MLP for RAM + action memory                                            │
│   • Combined latent vector                                                 │
│   • Policy head (actions)                                                  │
│   • Value head (state value)                                               │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼

┌────────────────────────────────────────────────────────────────────────────┐
│                       PPO Training Loop (Optimized)                        │
│                                                                            │
│   • n_steps = 128 (small rollouts)                                         │
│   • batch_size = 256                                                       │
│   • n_epochs = 4                                                           │
│   • gamma = 0.94                                                           │
│   • Learning rate: 2.5e-4 → 2.5e-6 (linear)                                │
│   • Clip range: 0.15 → 0.025 (linear)                                      │
│   • Monitor: clip_fraction, approx_kl, explained_variance                  │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼

┌────────────────────────────────────────────────────────────────────────────┐
│                            Checkpoints + Logs                              │
│                                                                            │
│   ./checkpoints_agent2/     → Auto-saved PPO checkpoints                    │
│   ./logs_agent2/            → TensorBoard metrics                           │
│                                                                            │
│   Track:                                                         │
│    • ep_rew_mean             • loss                                     │
│    • fps                     • explained_variance                        │
│    • clip_fraction           • policy_entropy                            │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼

┌────────────────────────────────────────────────────────────────────────────┐
│                            Final Agent Export                              │
│                                                                            │
│      agent2_final.zip  → Ready for DIAMBRA / 0G Hackathon submission      │
│                                                                            │
│   Run with:                                                                │
│      diambra run python agent.py                                           │
└────────────────────────────────────────────────────────────────────────────┘


# 🔴 Agent 1 — What Went Wrong

My first agent was functional but deeply inefficient.
Here are the mistakes and what I learned:

| Mistake             | What I Did                | What I Should Have Done                    |
| ------------------- | ------------------------- | ------------------------------------------ |
| **Wrappers**        | Built custom Gym wrappers | Use DIAMBRA’s optimized `WrappersSettings` |
| **Frame Resize**    | Python resize (slow)      | Engine-level resize (fast, C++)            |
| **Color Mode**      | RGB (3 channels)          | Grayscale (1 channel, 3× lighter)          |
| **Policy**          | `CnnPolicy`               | `MultiInputPolicy` (pixels + RAM)          |
| **Action Memory**   | ❌ None                    | 12-step action history (combos)            |
| **RAM Features**    | ❌ Ignored                 | Health, side, timer, stage, character      |
| **LR & Clip Range** | Static                    | Linear decay                               |
| **Gamma**           | 0.99                      | 0.94 (faster environments)                 |
| **Rollout Length**  | 2048                      | 128 (more responsive)                      |
| **Environments**    | Only 1                    | Multiple parallel envs                     |
| **FPS**             | 8                         | Target 15+                                 |

### Agent 1 Final Stats

```
Training Time: 31 hours
Total Timesteps: 1,001,472
FPS: 8
explained_variance: 0.905
clip_fraction: 0.51 (too high → unstable updates)
approx_kl: 0.039
```

Agent 1 trained, but inefficiently.
It became the baseline for the improved version.

---

# 🟢 Agent 2 — The Major Improvements

Agent 2 implements **best practices** found in DIAMBRA docs and fighting-game RL research.

### ✔ 1. Native DIAMBRA Wrappers

```python
from diambra.arena import WrappersSettings
wrappers_settings = WrappersSettings()
wrappers_settings.stack_frames = 4
wrappers_settings.add_last_action = True
wrappers_settings.stack_actions = 12
wrappers_settings.normalize_reward = True
```

### ✔ 2. Engine-level Frame Processing

```python
settings = EnvironmentSettings()
settings.frame_shape = (128, 128, 1)  # Grayscale
```

### ✔ 3. MultiInputPolicy (Pixels + RAM)

```python
model = PPO("MultiInputPolicy", env, ...)
wrappers_settings.filter_keys = [
    "action", "own_health", "opp_health",
    "own_side", "opp_side",
    "opp_character", "stage", "timer"
]
```

### ✔ 4. Linear Schedules

```python
def linear_schedule(start, end):
    return lambda p: end + p * (start - end)

learning_rate = linear_schedule(2.5e-4, 2.5e-6)
clip_range = linear_schedule(0.15, 0.025)
```

### ✔ 5. Fighting-Game PPO Hyperparameters

```python
gamma = 0.94
n_steps = 128
batch_size = 256
n_epochs = 4
```

### ✔ 6. Parallel Environments (Huge Speedup)

```
diambra run -s=6 python train_agent_v2_fast.py
```

---

# 📊 Agent 1 vs Agent 2

| Feature          | Agent 1             | Agent 2           |
| ---------------- | ------------------- | ----------------- |
| Frame Processing | Python, RGB         | Engine, Grayscale |
| Policy           | CnnPolicy           | MultiInputPolicy  |
| Frame Stack      | 4 via VecFrameStack | Native stack      |
| Action History   | ❌ None              | ✔ 12 actions      |
| RAM Features     | ❌ None              | ✔ 8 keys          |
| Reward Norm      | Custom              | Built-in          |
| LR Schedule      | Static              | Linear decay      |
| Gamma            | 0.99                | 0.94              |
| n_steps          | 2048                | 128               |
| Envs             | 1                   | 6                 |
| FPS              | 8                   | ~15               |
| Clip Fraction    | 0.51                | ~0.01             |
| approx_kl        | 0.039               | 0.001–0.002       |

Agent 2 learns **faster**, **more stably**, and **more efficiently**.

---

# 🧠 Key Lessons Learned

1. **Use DIAMBRA’s wrappers** — they are optimized for exactly this purpose.
2. **RAM states dramatically improve behavior**.
3. **Action memory is essential for combos**.
4. **Fighting games ≠ Atari** → different hyperparameters.
5. **Engine-side resizing is ~10× faster** than Python.
6. **Parallel training is the biggest speed boost**.
7. **Monitor clip_fraction** → >0.2 means bad updates.
8. **Documentation matters** — DIAMBRA provides great defaults.

---

# 🛠 Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

python -m venv venv
source venv/bin/activate

pip install diambra-arena[stable-baselines3]
pip install torch tensorboard

export DIAMBRAROMSPATH=/path/to/roms/
```

---

# 🚀 Usage

### Train Agent 2

```bash
diambra run -s=6 python train_agent_v2_fast.py
```

### Monitor Training

```
tensorboard --logdir=logs_agent2
```

### Run Trained Agent

```bash
diambra run python agent.py
```

---

# 📁 Project Structure

```
├── train_agent_v2_fast.py     # Optimized Agent 2 training
├── agent.py                   # Inference script
├── models_agent2/
│   └── agent2_final.zip
├── models/
│   └── ppo_sfiii3_final.zip   # Agent 1 baseline
├── checkpoints_agent2/
├── logs_agent2/
└── README.md
```

---

# 📄 Submission Instructions (0G)

1. Train your model.
2. Upload model folder:

   ```bash
   0g-storage upload models_agent2/
   ```
3. Submit the returned CID to the hackathon portal.

---

# 🏆 Hackathon Details

* Event: **DIAMBRA / 0G Cambridge AI Hackathon**
* Challenge: Train RL agents for Street Fighter III
* Submission: Via 0G Storage
* Finals: Live AI-vs-AI tournament

---

# 🙏 Acknowledgments

* **DIAMBRA Team** — for the RL fighting-game platform
* **0G Labs** — for decentralized model storage
* **Cambridge University** — for hosting the event
* **PyTorch / SB3 communities** — invaluable tools

---

# 📄 License

MIT License

---
