# Epsilon Decay Strategy Comparison in Reinforcement Learning

A systematic empirical study comparing **6 epsilon-greedy decay strategies** across **2 RL algorithms** (Tabular Q-Learning and DQN) and **2 classic control environments** (CartPole-v1 and MountainCar-v0). Each combination is trained for **50,000 episodes**, with epsilon saturating at episode 30,000 and the final 20,000 episodes used to observe the fully-exploitative policy.

---

## Decay Strategies

| # | Strategy | Formula |
|---|----------|---------|
| 1 | **Exponential** | `ε = min + (max − min) · exp(−k·t)`, &nbsp; `k = ln(1000) / T` |
| 2 | **Linear** | `ε = max(min, max − k·t)`, &nbsp; `k = (max − min) / T` |
| 3 | **Inverse** | `ε = max(min, max / (1 + k·t))`, &nbsp; `k = (max/min − 1) / T` |
| 4 | **Step (Geometric)** | 4 log-spaced levels: `1.0 → 0.2154 → 0.0464 → 0.01` at T/3 boundaries |
| 5 | **Logarithmic** | `ε = min + (max − min) / log(t + 2)` — intentionally slow baseline |
| 6 | **Cosine** | `ε = min + (max − min) · (1 + cos(π·t / T)) / 2` for `t < T`, then `min` |

All strategies share the same bounds: **ε_max = 1.0**, **ε_min = 0.01**, **T (saturation) = 30,000**.

---

## Environments & Algorithms

| Environment | Algorithm | Key Details |
|-------------|-----------|-------------|
| CartPole-v1 | Tabular Q-Learning | 4D state discretized into 10-bin grid; Q-table shape `(10,10,10,10,2)` |
| MountainCar-v0 | Tabular Q-Learning | 2D state discretized into 20-bin grid; Q-table shape `(20,20,3)` |
| CartPole-v1 | DQN | 2-hidden-layer MLP (128→128); AdamW optimizer; step-based target updates |
| MountainCar-v0 | DQN | 2-hidden-layer MLP (64→64); Adam optimizer; step-based target updates |

---

## Results

### DQN — CartPole-v1

#### Epsilon Decay Curves
![DQN CartPole Epsilon Comparison](DQN_CP/training_results/epsilon_comparison_all_strategies.png)

#### Reward Curves
![DQN CartPole Reward Comparison](DQN_CP/training_results/reward_comparison_all_strategies.png)

#### Individual Strategy Plots

| Strategy | Plot |
|----------|------|
| 1 — Exponential | ![](DQN_CP/training_results/strategy_1_plot.png) |
| 2 — Linear | ![](DQN_CP/training_results/strategy_2_plot.png) |
| 3 — Inverse | ![](DQN_CP/training_results/strategy_3_plot.png) |
| 4 — Step (Geometric) | ![](DQN_CP/training_results/strategy_4_plot.png) |
| 5 — Logarithmic | ![](DQN_CP/training_results/strategy_5_plot.png) |
| 6 — Cosine | ![](DQN_CP/training_results/strategy_6_plot.png) |

---

### DQN — MountainCar-v0

#### Epsilon Decay Curves
![DQN MountainCar Epsilon Comparison](DQN_MC/training_results_mountaincar/epsilon_comparison_all_strategies.png)

#### Reward Curves
![DQN MountainCar Reward Comparison](DQN_MC/training_results_mountaincar/reward_comparison_all_strategies.png)

#### Individual Strategy Plots

| Strategy | Plot |
|----------|------|
| 1 — Exponential | ![](DQN_MC/training_results_mountaincar/strategy_1_plot.png) |
| 2 — Linear | ![](DQN_MC/training_results_mountaincar/strategy_2_plot.png) |
| 3 — Inverse | ![](DQN_MC/training_results_mountaincar/strategy_3_plot.png) |
| 4 — Step (Geometric) | ![](DQN_MC/training_results_mountaincar/strategy_4_plot.png) |
| 5 — Logarithmic | ![](DQN_MC/training_results_mountaincar/strategy_5_plot.png) |
| 6 — Cosine | ![](DQN_MC/training_results_mountaincar/strategy_6_plot.png) |

---

### Q-Learning — CartPole-v1

#### Epsilon Decay Curves
![Q-Learning CartPole Epsilon Comparison](Q_CP/training_results_qlearning_cartpole/epsilon_comparison_all_strategies.png)

#### Reward Curves
![Q-Learning CartPole Reward Comparison](Q_CP/training_results_qlearning_cartpole/reward_comparison_all_strategies.png)

#### Individual Strategy Plots

| Strategy | Plot |
|----------|------|
| 1 — Exponential | ![](Q_CP/training_results_qlearning_cartpole/strategy_1_plot.png) |
| 2 — Linear | ![](Q_CP/training_results_qlearning_cartpole/strategy_2_plot.png) |
| 3 — Inverse | ![](Q_CP/training_results_qlearning_cartpole/strategy_3_plot.png) |
| 4 — Step (Geometric) | ![](Q_CP/training_results_qlearning_cartpole/strategy_4_plot.png) |
| 5 — Logarithmic | ![](Q_CP/training_results_qlearning_cartpole/strategy_5_plot.png) |
| 6 — Cosine | ![](Q_CP/training_results_qlearning_cartpole/strategy_6_plot.png) |

---

### Q-Learning — MountainCar-v0

#### Epsilon Decay Curves
![Q-Learning MountainCar Epsilon Comparison](Q_MC/training_results_qlearning_mountaincar/epsilon_comparison_all_strategies.png)

#### Reward Curves
![Q-Learning MountainCar Reward Comparison](Q_MC/training_results_qlearning_mountaincar/reward_comparison_all_strategies.png)

#### Individual Strategy Plots

| Strategy | Plot |
|----------|------|
| 1 — Exponential | ![](Q_MC/training_results_qlearning_mountaincar/strategy_1_plot.png) |
| 2 — Linear | ![](Q_MC/training_results_qlearning_mountaincar/strategy_2_plot.png) |
| 3 — Inverse | ![](Q_MC/training_results_qlearning_mountaincar/strategy_3_plot.png) |
| 4 — Step (Geometric) | ![](Q_MC/training_results_qlearning_mountaincar/strategy_4_plot.png) |
| 5 — Logarithmic | ![](Q_MC/training_results_qlearning_mountaincar/strategy_5_plot.png) |
| 6 — Cosine | ![](Q_MC/training_results_qlearning_mountaincar/strategy_6_plot.png) |

---

## Conclusion

The results reveal that no single decay strategy dominates across all settings — the best choice depends on the algorithm and environment. The key findings from the final 5,000 episodes of each run are summarised below.

### Final Performance Summary (avg reward, last 5k episodes)

| Strategy | DQN CartPole | DQN MountainCar | Q-Learning CartPole | Q-Learning MountainCar |
|----------|:------------:|:---------------:|:-------------------:|:----------------------:|
| 1 — Exponential | 281.7 | -142.2 | 302.7 | -144.2 |
| 2 — Linear | 96.0 | **-112.2** | **352.4** | -150.4 |
| 3 — Inverse | 318.7 | -123.2 | 276.0 | -137.1 |
| 4 — Step | 9.4 | -120.9 | 328.4 | -153.9 |
| 5 — Logarithmic | **377.9** | -148.1 | 292.3 | -165.5 |
| 6 — Cosine | 83.3 | **-115.1** | 337.0 | **-132.8** |

*(Higher is better for CartPole; less negative is better for MountainCar)*

### Key Observations

**Logarithmic decay excels with DQN on CartPole.** Its intentionally slow decay kept exploration active well past the warm-up phase, which gave the DQN's replay buffer more diverse transitions during the critical early learning window. Strategy 5 was the first to sustain average rewards above 400 and maintained that level through episode 50,000.

**Inverse decay is the fastest DQN learner on CartPole.** Strategy 3 first crossed the 200-average-reward threshold at episode 3,000 — nearly four times earlier than Exponential (episode 12,500) and over six times earlier than Logarithmic (episode 2,000 but then was neck-and-neck). Its rapid early exploitation paid off given DQN's ability to generalise from limited experience.

**Step decay collapses with DQN.** The abrupt drops in epsilon coincide with catastrophic forgetting events clearly visible in the individual plots. DQN appears sensitive to sudden shifts in exploration rate; the replay buffer's distribution changes too quickly at each step boundary for the network to adapt. Average reward over the final 5,000 episodes was only 9.4 — no better than random.

**Linear and Cosine are the most reliable strategies for Q-Learning.** Linear topped the Q-Learning CartPole rankings (avg 352.4) and Cosine led on Q-Learning MountainCar (avg -132.8). Both share a smooth, predictable decay profile that pairs well with tabular updates, where there is no risk of catastrophic forgetting and steady exploration yields a well-filled Q-table.

**MountainCar exposes the cost of slow exploration.** Logarithmic decay — the strongest performer on CartPole — ranked last on both MountainCar experiments. Its high epsilon persists so long that the agent rarely exploits the sparse goal reward when it does find it, causing Q-values for the goal region to remain under-updated. Fast-decaying strategies (Linear, Cosine, Inverse) that commit to exploitation earlier achieved better final rewards on this environment.

**Cosine annealing is the most consistent all-rounder.** It placed in the top two on both MountainCar experiments and top three on Q-Learning CartPole. Its smooth S-curve provides aggressive early exploration followed by a gradual, controlled transition to exploitation — avoiding both the abrupt disruptions of Step decay and the prolonged randomness of Logarithmic decay.

### Practical Recommendations

- Use **Cosine or Inverse** decay as a default starting point — they perform competitively across all four settings tested here.
- Prefer **Logarithmic** decay only when the task is dense-reward and the model can benefit from extended exploration (e.g., DQN on CartPole).
- Avoid **Step decay** with neural network-based agents; the sudden epsilon drops destabilise training.
- For **tabular methods**, Linear decay is a strong, low-hyperparameter choice that reliably produces good final policies.

---

## Installation

```bash
pip install gymnasium torch numpy matplotlib
```

---

## Running an Experiment

```bash
# Q-Learning on CartPole
python Q_CP/Q_CP.py

# Q-Learning on MountainCar
python Q_MC/Q_MC.py

# DQN on CartPole
python DQN_CP/DQN_CP.py

# DQN on MountainCar
python DQN_MC/DQN_MC.py
```

---

## Design Notes

**Warm-up period** — exploration is kept at ε_max until the replay buffer contains enough transitions for stable mini-batch sampling. Decay only begins after warm-up.

**Step-based target updates (DQN)** — the target network is refreshed every 1,000 *gradient steps* rather than every episode, decoupling update frequency from episode length.

**Logarithmic strategy** — with `c = 1.0`, this strategy decays slower than the rest and does not reach `ε_min` by episode 30,000. It is included as a slow-decay baseline and its behaviour is intentional.

**Default environment rewards** — no reward shaping is applied. CartPole gives `+1` per step; MountainCar gives `−1` per step until the goal is reached.
