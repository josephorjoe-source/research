import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import os
import csv

# ---------------------------------------------------------
# Hyperparameters & Setup
# ---------------------------------------------------------
EPISODES = 50000
SATURATION_EPISODE = 30000  # Decay ends here; last 20k episodes observe saturated model
WARMUP_EPISODES = 64  # Warm-up: keep epsilon=MAX while buffer fills

MAX_EPSILON = 1.0
MIN_EPSILON = 0.01

BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4  # FIXED: Changed from 1e-3 to 1e-4 (more stable)
TARGET_UPDATE = 1000  # FIXED: Now based on STEPS, not episodes
MEMORY_SIZE = 50000

# Create results directory
RESULTS_DIR = "training_results_mountaincar"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------
# Epsilon Helper Function
# ---------------------------------------------------------
def get_epsilon(strategy, episode, max_epsilon, min_epsilon, decay_rate, saturation_episode, c=1.0):
    """
    All strategies are tuned so that epsilon reaches min_epsilon at saturation_episode (T=30k).
    The remaining 20k episodes stay at min_epsilon for saturated-model observation.
    
    NOTE ON WARM-UP: DQN benefits from a warm-up window (~WARMUP_EPISODES) during which the
    replay buffer fills. We pass an *adjusted* episode index (episode - WARMUP_EPISODES) so
    that decay only begins after the buffer is ready. The caller handles this offset.
    
    Strategy 1 — Exponential:   ε = min + (max-min) * exp(-k*t)
                                 k = ln(1000) / T  →  ε(T) ≈ 0.001*(max-min)+min ≈ min_epsilon
                                 (exact zero is unreachable; 0.1 % residual is the standard proxy)
    
    Strategy 2 — Linear:        ε = max(min, max - k*t)
                                 k = (max - min) / T  →  ε(T) = min_epsilon exactly
    
    Strategy 3 — Inverse:       ε = max(min, max / (1 + k*t))
                                 k = (max/min - 1) / T  →  ε(T) = min_epsilon exactly
    
    Strategy 4 — Step (geometric):
                                 4 levels equally spaced in log-space between max and min.
                                 Boundaries at T/3, 2T/3, T so the last step lands exactly
                                 at min_epsilon. Steps: 1.0 → ~0.2154 → ~0.0464 → 0.01
    
    Strategy 5 — Logarithmic:   ε = min + (max-min) / log(t + 1 + c)
                                 Log decay is inherently slow. Exact saturation at T requires
                                 c = exp((max-min)/min_epsilon) - T - 1, which is astronomical.
                                 Instead c = 1.0 is kept (honest slow-decay behaviour) and this
                                 strategy serves as the slowest-decay baseline in the comparison.
    
    Strategy 6 — Cosine:        ε = min + (max-min) * (1 + cos(π*t / T)) / 2,  t < T
                                 Flatlines at min_epsilon for t >= T (the 20k observation window).
                                 saturation_episode (not total_episodes) is the cosine period.
    """
    if strategy == 1:
        return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    elif strategy == 2:
        return max(min_epsilon, max_epsilon - decay_rate * episode)
    elif strategy == 3:
        return max(min_epsilon, max_epsilon / (1 + decay_rate * episode))
    elif strategy == 4:
        # Geometric steps in log-space: r = (min/max)^(1/3)
        # Boundaries at T/3 ≈ 10000, 2T/3 ≈ 20000, T = 30000
        r = (min_epsilon / max_epsilon) ** (1 / 3)  # ≈ 0.2154
        if   episode < 10000: return max_epsilon          # 1.0
        elif episode < 20000: return max_epsilon * r      # ≈ 0.2154
        elif episode < 30000: return max_epsilon * r ** 2 # ≈ 0.0464
        else:                 return min_epsilon           # 0.01
    elif strategy == 5:
        # c = 1.0: slow logarithmic curve, acknowledged research baseline
        return min_epsilon + (max_epsilon - min_epsilon) / np.log(episode + 1 + c)
    elif strategy == 6:
        # Cosine period = saturation_episode (30k), NOT total_episodes (50k)
        if episode >= saturation_episode:
            return min_epsilon
        return min_epsilon + (max_epsilon - min_epsilon) * (1 + np.cos(np.pi * episode / saturation_episode)) / 2

# ---------------------------------------------------------
# Neural Network & Replay Buffer
# ---------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)

# ---------------------------------------------------------
# Decay Rate Calculation
# ---------------------------------------------------------
T = SATURATION_EPISODE

decay_rates = {
    1: np.log(1000) / T,                    # ≈ 0.0002303
    2: (MAX_EPSILON - MIN_EPSILON) / T,      # ≈ 0.000033
    3: (MAX_EPSILON / MIN_EPSILON - 1) / T,  # ≈ 0.0033
    4: None,
    5: None,
    6: None,
}

LOG_C = 1.0  # Strategy 5 only

# ---------------------------------------------------------
# Training Function for a Single Strategy
# ---------------------------------------------------------
def train_strategy(strategy_num):
    print(f"\n{'='*60}")
    print(f"Starting Training: Strategy {strategy_num} | Episodes {EPISODES}")
    print(f"{'='*60}\n")
    
    # Environment setup
    env = gym.make('MountainCar-v0')
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Network setup
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory    = ReplayBuffer(MEMORY_SIZE)

    # Data tracking
    ep_rewards      = []
    epsilon_history = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
    
    # FIXED: Step counter for target network updates
    steps_done = 0
    
    DECAY_RATE = decay_rates[strategy_num]

    # ---------------------------------------------------------
    # Main Training Loop
    # ---------------------------------------------------------
    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done         = False

        # Offset episode index so decay only begins after warm-up.
        # During warm-up, episode_for_decay stays at 0 → epsilon = MAX_EPSILON.
        episode_for_decay = max(0, episode - WARMUP_EPISODES)
        epsilon = get_epsilon(
            strategy_num, episode_for_decay,
            MAX_EPSILON, MIN_EPSILON,
            DECAY_RATE, SATURATION_EPISODE,
            c=LOG_C
        )
        epsilon_history.append(epsilon)

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # FIXED: Use default MountainCar reward (no custom shaping)
            # Optional shaped reward (uncomment to accelerate learning):
            # reward = reward + abs(next_state[0] - (-0.5))

            memory.push(state, action, reward, next_state, done)
            state         = next_state
            total_reward += reward

            # Only optimise once the buffer has enough samples
            if len(memory) > BATCH_SIZE:
                states_b, actions_b, rewards_b, next_states_b, dones_b = memory.sample(BATCH_SIZE)
                states_b      = torch.FloatTensor(states_b).to(device)
                actions_b     = torch.LongTensor(actions_b).unsqueeze(1).to(device)
                rewards_b     = torch.FloatTensor(rewards_b).unsqueeze(1).to(device)
                next_states_b = torch.FloatTensor(next_states_b).to(device)
                dones_b       = torch.FloatTensor(dones_b).unsqueeze(1).to(device)

                q_values      = policy_net(states_b).gather(1, actions_b)
                next_q_values = target_net(next_states_b).max(1)[0].unsqueeze(1)
                target_q      = rewards_b + (GAMMA * next_q_values * (1 - dones_b))

                loss = nn.MSELoss()(q_values, target_q.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # FIXED: Update target network every TARGET_UPDATE steps (not episodes)
                steps_done += 1
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        ep_rewards.append(total_reward)
        
        if not episode % 500:
            avg_reward = sum(ep_rewards[-500:]) / len(ep_rewards[-500:])
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(avg_reward)
            aggr_ep_rewards['min'].append(min(ep_rewards[-500:]))
            aggr_ep_rewards['max'].append(max(ep_rewards[-500:]))
            print(f"Episode: {episode:>5d}, avg: {avg_reward:>4.1f}, min: {min(ep_rewards[-500:]):>4.1f}, max: {max(ep_rewards[-500:]):>4.1f}, epsilon: {epsilon:>1.5f}, steps: {steps_done}")

    env.close()

    # ---------------------------------------------------------
    # Save Model
    # ---------------------------------------------------------
    model_path = os.path.join(RESULTS_DIR, f"dqn_mountaincar_strategy_{strategy_num}.pth")
    torch.save(policy_net.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # ---------------------------------------------------------
    # Save CSV with Rewards and Episodes
    # ---------------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, f"strategy_{strategy_num}_rewards.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Reward', 'Epsilon'])
        for ep_num, (reward, epsilon) in enumerate(zip(ep_rewards, epsilon_history)):
            writer.writerow([ep_num, reward, epsilon])
    print(f"Rewards CSV saved to {csv_path}")

    # ---------------------------------------------------------
    # Save Aggregated CSV (500-episode windows)
    # ---------------------------------------------------------
    agg_csv_path = os.path.join(RESULTS_DIR, f"strategy_{strategy_num}_aggregated.csv")
    with open(agg_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Avg_Reward', 'Min_Reward', 'Max_Reward'])
        for ep, avg, min_r, max_r in zip(aggr_ep_rewards['ep'], 
                                          aggr_ep_rewards['avg'], 
                                          aggr_ep_rewards['min'], 
                                          aggr_ep_rewards['max']):
            writer.writerow([ep, avg, min_r, max_r])
    print(f"Aggregated CSV saved to {agg_csv_path}")

    # ---------------------------------------------------------
    # Create and Save Plot
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Plot 1: Rewards (avg/min/max per 500-episode window)
    ax1.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='Average Reward', color='blue')
    ax1.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='Min Reward',     color='red',   alpha=0.3)
    ax1.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='Max Reward',     color='green', alpha=0.3)
    ax1.axvline(x=SATURATION_EPISODE, color='black', linestyle='--', alpha=0.5,
                label='Target Saturation (Ep 30k)')
    ax1.axvline(x=WARMUP_EPISODES,    color='grey',  linestyle=':',  alpha=0.5,
                label=f'Warm-up End (Ep {WARMUP_EPISODES})')
    ax1.set_title(f'DQN MountainCar: Rewards vs Episodes (Strategy {strategy_num} - 50k Episodes)')
    ax1.set_ylabel('Reward')
    ax1.legend(loc='best')
    ax1.grid(True)

    # Plot 2: Epsilon decay curve
    ax2.plot(range(EPISODES), epsilon_history, color='purple', label='Epsilon Value')
    ax2.axvline(x=SATURATION_EPISODE, color='black', linestyle='--', alpha=0.5,
                label='Target Saturation (Ep 30k)')
    ax2.axvline(x=WARMUP_EPISODES,    color='grey',  linestyle=':',  alpha=0.5,
                label=f'Warm-up End (Ep {WARMUP_EPISODES})')
    ax2.axhline(y=MIN_EPSILON,        color='orange', linestyle=':',  alpha=0.7,
                label=f'Min Epsilon ({MIN_EPSILON})')
    ax2.set_title(f'Epsilon Decay Curve (Strategy {strategy_num})')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Epsilon')
    ax2.legend(loc='best')
    ax2.grid(True)

    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(RESULTS_DIR, f"strategy_{strategy_num}_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.close()
    
    return {
        'strategy': strategy_num,
        'episode_rewards': ep_rewards,
        'epsilon_history': epsilon_history,
        'aggr_ep_rewards': aggr_ep_rewards
    }

# ---------------------------------------------------------
# Main Execution: Loop Through All Strategies
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"# DQN MountainCar - Multi-Strategy Training (FIXED VERSION)")
    print(f"# FIXES APPLIED:")
    print(f"#   - Learning Rate: {LR} (reduced from 1e-3)")
    print(f"#   - Target Updates: Every {TARGET_UPDATE} STEPS (not episodes)")
    print(f"#   - Rewards: Using default MountainCar rewards")
    print(f"# Testing Strategies 1-6")
    print(f"# Episodes: {EPISODES}")
    print(f"# Results will be saved to: {RESULTS_DIR}/")
    print(f"{'#'*60}\n")
    
    all_results = {}
    
    # Train each strategy
    for strategy in range(1, 7):
        result = train_strategy(strategy)
        all_results[strategy] = result
        print(f"\n✓ Strategy {strategy} completed!\n")
    
    # ---------------------------------------------------------
    # Create Comparison Plots
    # ---------------------------------------------------------
    print(f"\n{'='*60}")
    print("Creating comparison plots...")
    print(f"{'='*60}\n")
    
    strategy_names = {
        1: 'Exponential',
        2: 'Linear',
        3: 'Inverse',
        4: 'Step (Geometric)',
        5: 'Logarithmic',
        6: 'Cosine'
    }
    
    # Comparison Plot: All Epsilon Curves
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, strategy in enumerate(range(1, 7)):
        ax.plot(range(EPISODES), all_results[strategy]['epsilon_history'], 
                label=f"Strategy {strategy}: {strategy_names[strategy]}", 
                color=colors[i], alpha=0.8)
    
    ax.axvline(x=SATURATION_EPISODE, color='black', linestyle='--', alpha=0.5, label='Target Saturation')
    ax.axvline(x=WARMUP_EPISODES, color='grey', linestyle=':', alpha=0.5, label='Warm-up End')
    ax.axhline(y=MIN_EPSILON, color='red', linestyle=':', alpha=0.5, label=f'Min Epsilon ({MIN_EPSILON})')
    
    ax.set_title('Epsilon Decay Comparison: All Strategies (MountainCar)')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Epsilon')
    ax.legend(loc='best')
    ax.grid(True)
    
    comparison_path = os.path.join(RESULTS_DIR, "epsilon_comparison_all_strategies.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Epsilon comparison plot saved to {comparison_path}")
    plt.close()
    
    # Comparison Plot: All Reward Curves
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    for i, strategy in enumerate(range(1, 7)):
        aggr = all_results[strategy]['aggr_ep_rewards']
        ax.plot(aggr['ep'], aggr['avg'], 
                label=f"Strategy {strategy}: {strategy_names[strategy]}", 
                color=colors[i], alpha=0.8, linewidth=2)
    
    ax.axvline(x=SATURATION_EPISODE, color='black', linestyle='--', alpha=0.5, label='Target Saturation')
    ax.axvline(x=WARMUP_EPISODES, color='grey', linestyle=':', alpha=0.5, label='Warm-up End')
    
    ax.set_title('Average Reward Comparison: All Strategies (500-episode windows) - MountainCar')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Average Reward')
    ax.legend(loc='best')
    ax.grid(True)
    
    reward_comparison_path = os.path.join(RESULTS_DIR, "reward_comparison_all_strategies.png")
    plt.savefig(reward_comparison_path, dpi=300, bbox_inches='tight')
    print(f"Reward comparison plot saved to {reward_comparison_path}")
    plt.close()
    
    print(f"\n{'='*60}")
    print("✓ ALL TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"\nAll results saved to: {RESULTS_DIR}/")
    print("\nGenerated files:")
    print("  - 6 model files (.pth)")
    print("  - 6 detailed reward CSVs (per episode)")
    print("  - 6 aggregated CSVs (500-episode windows)")
    print("  - 6 individual strategy plots")
    print("  - 2 comparison plots (epsilon & rewards)")
    print(f"\n{'='*60}\n")