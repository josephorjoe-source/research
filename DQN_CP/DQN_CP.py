import gymnasium as gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os
import csv

# ---------------------------------------------------------
# Hyperparameters & Setup
# ---------------------------------------------------------
EPISODES = 50000
SATURATION_EPISODE = 30000
WARMUP_EPISODES = 1000
MAX_EPSILON = 1.0
MIN_EPSILON = 0.01

BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4  # FIXED: Changed from 1e-3 to 1e-4 (more stable)
TARGET_UPDATE = 1000  # FIXED: Now based on STEPS, not episodes
MEMORY_SIZE = 10000

# Create results directory
RESULTS_DIR = "training_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Device configuration for Mac (Apple Silicon MPS or CPU)
device = "cpu" #torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------
# Epsilon Decay Strategies
# ---------------------------------------------------------
def get_epsilon(strategy, t, T=SATURATION_EPISODE, max_e=MAX_EPSILON, min_e=MIN_EPSILON):
    if strategy == 1:
        # Exponential
        k = math.log(1000) / T
        return min_e + (max_e - min_e) * math.exp(-k * t)
    
    elif strategy == 2:
        # Linear
        k = (max_e - min_e) / T
        return max(min_e, max_e - k * t)
    
    elif strategy == 3:
        # Inverse
        k = (max_e / min_e - 1) / T
        return max(min_e, max_e / (1 + k * t))
    
    elif strategy == 4:
        # Step (geometric)
        levels = np.geomspace(max_e, min_e, 4)
        if t < T / 3:
            return levels[0]
        elif t < 2 * T / 3:
            return levels[1]
        elif t < T:
            return levels[2]
        else:
            return levels[3]
            
    elif strategy == 5:
        # Logarithmic
        # Capped at max_e because for t=0, log(2) < 1, causing epsilon > 1.0
        val = min_e + (max_e - min_e) / math.log(t + 1 + 1.0)
        return min(max_e, val)
        
    elif strategy == 6:
        # Cosine
        if t < T:
            return min_e + (max_e - min_e) * (1 + math.cos(math.pi * t / T)) / 2
        else:
            return min_e
    else:
        raise ValueError("Invalid strategy number. Choose 1-6.")

# ---------------------------------------------------------
# DQN Architecture & Replay Buffer
# ---------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# ---------------------------------------------------------
# Training Function for a Single Strategy
# ---------------------------------------------------------
def train_strategy(strategy_num):
    print(f"\n{'='*60}")
    print(f"Starting Training: Strategy {strategy_num} | Episodes {EPISODES}")
    print(f"{'='*60}\n")
    
    # Environment setup
    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    # Network setup
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_SIZE)

    # Data tracking for plots and CSV
    epsilon_history = []
    episode_rewards = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
    window_rewards = []
    
    # FIXED: Step counter for target network updates
    steps_done = 0

    # ---------------------------------------------------------
    # Main Training Loop
    # ---------------------------------------------------------
    for ep in range(EPISODES):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        ep_reward = 0
        eps_threshold = get_epsilon(strategy_num, ep)
        epsilon_history.append(eps_threshold)

        while True:
            # Action Selection
            if ep < WARMUP_EPISODES or random.random() < eps_threshold:
                action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
            else:
                with torch.no_grad():
                    action = policy_net(state).max(1)[1].view(1, 1)

            observation, reward, terminated, truncated, _ = env.step(action.item())
            ep_reward += reward
            done = terminated or truncated
            
            # FIXED: Use default CartPole reward (no custom penalty)
            reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)

            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) if not done else None

            # Store in memory
            memory.push(state, action, reward_tensor, next_state, torch.tensor([done], device=device, dtype=torch.bool))
            state = next_state

            # Optimization Step
            if len(memory) >= BATCH_SIZE and ep >= WARMUP_EPISODES:
                transitions = memory.sample(BATCH_SIZE)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                batch_state = torch.cat(batch_state)
                batch_action = torch.cat(batch_action)
                batch_reward = torch.cat(batch_reward)
                batch_done = torch.cat(batch_done)

                non_final_mask = ~batch_done
                non_final_next_states = torch.cat([s for s in batch_next_state if s is not None])

                state_action_values = policy_net(batch_state).gather(1, batch_action)

                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                with torch.no_grad():
                    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

                expected_state_action_values = (next_state_values * GAMMA) + batch_reward

                criterion = nn.SmoothL1Loss()
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                optimizer.step()
                
                # FIXED: Update target network every TARGET_UPDATE steps (not episodes)
                steps_done += 1
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # Tracking metrics
        episode_rewards.append(ep_reward)
        window_rewards.append(ep_reward)
        
        # Aggregate data every 500 episodes for plotting
        if (ep + 1) % 500 == 0:
            aggr_ep_rewards['ep'].append(ep + 1)
            aggr_ep_rewards['avg'].append(np.mean(window_rewards))
            aggr_ep_rewards['min'].append(np.min(window_rewards))
            aggr_ep_rewards['max'].append(np.max(window_rewards))
            print(f"Episode {ep+1} | Avg Reward: {np.mean(window_rewards):.2f} | Epsilon: {eps_threshold:.4f} | Steps: {steps_done}")
            window_rewards = []

    env.close()

    # ---------------------------------------------------------
    # Save Model
    # ---------------------------------------------------------
    model_path = os.path.join(RESULTS_DIR, f"dqn_cartpole_strategy_{strategy_num}.pth")
    torch.save(policy_net.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # ---------------------------------------------------------
    # Save CSV with Rewards and Episodes
    # ---------------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, f"strategy_{strategy_num}_rewards.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Reward', 'Epsilon'])
        for ep_num, (reward, epsilon) in enumerate(zip(episode_rewards, epsilon_history), start=1):
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
    ax1.set_title(f'DQN: Rewards vs Episodes (Strategy {strategy_num} - 50k Episodes)')
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
        'episode_rewards': episode_rewards,
        'epsilon_history': epsilon_history,
        'aggr_ep_rewards': aggr_ep_rewards
    }

# ---------------------------------------------------------
# Main Execution: Loop Through All Strategies
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"# DQN CartPole - Multi-Strategy Training (FIXED VERSION)")
    print(f"# FIXES APPLIED:")
    print(f"#   - Learning Rate: 1e-4 (reduced from 1e-3)")
    print(f"#   - Target Updates: Every {TARGET_UPDATE} STEPS (not episodes)")
    print(f"#   - Rewards: Using default CartPole rewards (no penalty)")
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
    # Create Comparison Plot
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
    
    ax.set_title('Epsilon Decay Comparison: All Strategies')
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
    
    ax.set_title('Average Reward Comparison: All Strategies (500-episode windows)')
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