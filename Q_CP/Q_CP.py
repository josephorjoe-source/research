import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import csv

# ---------------------------------------------------------
# Hyperparameters & Setup
# ---------------------------------------------------------
TOTAL_EPISODES = 50000
SATURATION_EPISODE = 30000  # T
MAX_EPS = 1.0
MIN_EPS = 0.01

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
BINS = [10, 10, 10, 10]  # Discretization bins for the 4 CartPole observations
WINDOW = 500
WARMUP_EPISODES = 1000

# Create results directory
RESULTS_DIR = "training_results_qlearning_cartpole"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# Epsilon Decay Functions
# ---------------------------------------------------------
def eps_exponential(t):
    k = math.log(1000) / SATURATION_EPISODE
    return MIN_EPS + (MAX_EPS - MIN_EPS) * math.exp(-k * t)

def eps_linear(t):
    k = (MAX_EPS - MIN_EPS) / SATURATION_EPISODE
    return max(MIN_EPS, MAX_EPS - k * t)

def eps_inverse(t):
    k = ((MAX_EPS / MIN_EPS) - 1) / SATURATION_EPISODE
    return max(MIN_EPS, MAX_EPS / (1 + k * t))

def eps_step(t):
    # Calculate geometric steps dynamically
    steps = np.geomspace(MAX_EPS, MIN_EPS, 4)
    if t < SATURATION_EPISODE / 3:
        return steps[0]
    elif t < 2 * SATURATION_EPISODE / 3:
        return steps[1]
    elif t < SATURATION_EPISODE:
        return steps[2]
    else:
        return steps[3]

def eps_logarithmic(t):
    # c = 1.0, so log(t + 1 + 1) -> log(t + 2)
    # Capped at MAX_EPS because log(2) yields a multiplier > 1
    val = MIN_EPS + (MAX_EPS - MIN_EPS) / math.log(t + 2)
    return min(MAX_EPS, val)

def eps_cosine(t):
    if t < SATURATION_EPISODE:
        return MIN_EPS + (MAX_EPS - MIN_EPS) * (1 + math.cos(math.pi * t / SATURATION_EPISODE)) / 2
    else:
        return MIN_EPS

# ---------------------------------------------------------
# Environment Setup & Discretization
# ---------------------------------------------------------
def create_bins():
    # CartPole observation limits: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
    return [
        np.linspace(-2.4, 2.4, BINS[0] - 1),
        np.linspace(-3.0, 3.0, BINS[1] - 1),
        np.linspace(-0.209, 0.209, BINS[2] - 1),
        np.linspace(-3.0, 3.0, BINS[3] - 1)
    ]

def discretize_state(state, bins):
    state_indices = []
    for i in range(len(state)):
        state_indices.append(np.digitize(state[i], bins[i]))
    return tuple(state_indices)

# ---------------------------------------------------------
# Q-Learning Training Loop
# ---------------------------------------------------------
def train_agent(decay_strategy, strategy_name, strategy_num):
    print(f"\n{'='*60}")
    print(f"Starting Training: Strategy {strategy_num} - {strategy_name} | Episodes {TOTAL_EPISODES}")
    print(f"{'='*60}\n")
    
    env = gym.make("CartPole-v1")
    bins = create_bins()
    
    # Initialize Q-table: shape is (10, 10, 10, 10, 2)
    q_table = np.zeros(BINS + [env.action_space.n])
    
    rewards_per_episode = np.zeros(TOTAL_EPISODES)
    eps_history = np.zeros(TOTAL_EPISODES)
    
    for episode in range(TOTAL_EPISODES):
        state, _ = env.reset()
        state = discretize_state(state, bins)
        done = False
        truncated = False
        total_reward = 0
        
        epsilon = decay_strategy(episode)
        eps_history[episode] = epsilon
        
        while not (done or truncated):
            # Epsilon-Greedy Action Selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
                
            next_state, reward, done, truncated, _ = env.step(action)
            next_state_discrete = discretize_state(next_state, bins)
            
            # Q-Learning Update
            best_next_action = np.argmax(q_table[next_state_discrete])
            td_target = reward + DISCOUNT_FACTOR * q_table[next_state_discrete][best_next_action] * (not (done or truncated))
            td_error = td_target - q_table[state][action]
            
            q_table[state][action] += LEARNING_RATE * td_error
            
            state = next_state_discrete
            total_reward += reward
            
        rewards_per_episode[episode] = total_reward
        
        # Progress tracker
        if (episode + 1) % 10000 == 0:
            avg_last_1k = np.mean(rewards_per_episode[episode - 999:episode + 1])
            print(f"Episode {episode + 1}/{TOTAL_EPISODES} | Avg reward (last 1k): {avg_last_1k:.2f} | Epsilon: {epsilon:.4f}")
            
    env.close()
    
    # ---------------------------------------------------------
    # Save Model
    # ---------------------------------------------------------
    model_path = os.path.join(RESULTS_DIR, f"qtable_cartpole_strategy_{strategy_num}_{strategy_name}.npy")
    np.save(model_path, q_table)
    print(f"\nModel saved to {model_path}")
    
    # ---------------------------------------------------------
    # Save CSV with Rewards and Episodes
    # ---------------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, f"strategy_{strategy_num}_rewards.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Reward', 'Epsilon'])
        for ep_num, (reward, epsilon) in enumerate(zip(rewards_per_episode, eps_history)):
            writer.writerow([ep_num, reward, epsilon])
    print(f"Rewards CSV saved to {csv_path}")
    
    # ---------------------------------------------------------
    # Aggregate Data for Windowed Plot
    # ---------------------------------------------------------
    episodes_aggr = np.arange(0, TOTAL_EPISODES, WINDOW)
    avg_rewards = []
    min_rewards = []
    max_rewards = []
    
    for i in episodes_aggr:
        window_slice = rewards_per_episode[i : i + WINDOW]
        avg_rewards.append(np.mean(window_slice))
        min_rewards.append(np.min(window_slice))
        max_rewards.append(np.max(window_slice))
    
    # ---------------------------------------------------------
    # Save Aggregated CSV (500-episode windows)
    # ---------------------------------------------------------
    agg_csv_path = os.path.join(RESULTS_DIR, f"strategy_{strategy_num}_aggregated.csv")
    with open(agg_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Avg_Reward', 'Min_Reward', 'Max_Reward'])
        for ep, avg, min_r, max_r in zip(episodes_aggr, avg_rewards, min_rewards, max_rewards):
            writer.writerow([ep, avg, min_r, max_r])
    print(f"Aggregated CSV saved to {agg_csv_path}")
    
    # ---------------------------------------------------------
    # Create and Save Plot
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Plot 1: Rewards (avg/min/max per window)
    ax1.plot(episodes_aggr, avg_rewards, label='Average Reward', color='blue')
    ax1.plot(episodes_aggr, min_rewards, label='Min Reward',     color='red',   alpha=0.3)
    ax1.plot(episodes_aggr, max_rewards, label='Max Reward',     color='green', alpha=0.3)
    
    ax1.axvline(x=SATURATION_EPISODE, color='black', linestyle='--', alpha=0.5,
                label='Target Saturation (Ep 30k)')
    ax1.axvline(x=WARMUP_EPISODES,    color='grey',  linestyle=':',  alpha=0.5,
                label=f'Warm-up End (Ep {WARMUP_EPISODES})')
    
    ax1.set_title(f'Tabular Q-Learning CartPole: Rewards vs Episodes (Strategy {strategy_num}: {strategy_name})')
    ax1.set_ylabel('Reward')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.4)

    # Plot 2: Epsilon decay curve
    ax2.plot(range(TOTAL_EPISODES), eps_history, color='purple', label='Epsilon Value')
    
    ax2.axvline(x=SATURATION_EPISODE, color='black', linestyle='--', alpha=0.5,
                label='Target Saturation (Ep 30k)')
    ax2.axvline(x=WARMUP_EPISODES,    color='grey',  linestyle=':',  alpha=0.5,
                label=f'Warm-up End (Ep {WARMUP_EPISODES})')
    ax2.axhline(y=MIN_EPS,            color='orange', linestyle=':',  alpha=0.7,
                label=f'Min Epsilon ({MIN_EPS})')
    
    ax2.set_title(f'Epsilon Decay Curve (Strategy {strategy_num}: {strategy_name})')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Epsilon')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(RESULTS_DIR, f"strategy_{strategy_num}_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.close()
    
    return {
        'strategy': strategy_num,
        'strategy_name': strategy_name,
        'rewards': rewards_per_episode,
        'epsilon_history': eps_history,
        'aggr_episodes': episodes_aggr,
        'avg_rewards': avg_rewards,
        'min_rewards': min_rewards,
        'max_rewards': max_rewards
    }

# ---------------------------------------------------------
# Main Execution: Loop Through All Strategies
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"# Q-Learning CartPole - Multi-Strategy Training")
    print(f"# Testing Strategies 1-6")
    print(f"# Episodes: {TOTAL_EPISODES}")
    print(f"# Results will be saved to: {RESULTS_DIR}/")
    print(f"{'#'*60}\n")
    
    strategies = {
        1: ("Exponential", eps_exponential),
        2: ("Linear", eps_linear),
        3: ("Inverse", eps_inverse),
        4: ("Step_Geometric", eps_step),
        5: ("Logarithmic", eps_logarithmic),
        6: ("Cosine", eps_cosine)
    }
    
    all_results = {}
    
    # Train each strategy
    for strategy_num, (strategy_name, decay_func) in strategies.items():
        result = train_agent(decay_func, strategy_name, strategy_num)
        all_results[strategy_num] = result
        print(f"\n✓ Strategy {strategy_num} ({strategy_name}) completed!\n")
    
    # ---------------------------------------------------------
    # Create Comparison Plots
    # ---------------------------------------------------------
    print(f"\n{'='*60}")
    print("Creating comparison plots...")
    print(f"{'='*60}\n")
    
    # Comparison Plot: All Epsilon Curves
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, strategy_num in enumerate(range(1, 7)):
        result = all_results[strategy_num]
        ax.plot(range(TOTAL_EPISODES), result['epsilon_history'], 
                label=f"Strategy {strategy_num}: {result['strategy_name']}", 
                color=colors[i], alpha=0.8)
    
    ax.axvline(x=SATURATION_EPISODE, color='black', linestyle='--', alpha=0.5, label='Target Saturation')
    ax.axvline(x=WARMUP_EPISODES, color='grey', linestyle=':', alpha=0.5, label='Warm-up End')
    ax.axhline(y=MIN_EPS, color='red', linestyle=':', alpha=0.5, label=f'Min Epsilon ({MIN_EPS})')
    
    ax.set_title('Epsilon Decay Comparison: All Strategies (Q-Learning CartPole)')
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
    
    for i, strategy_num in enumerate(range(1, 7)):
        result = all_results[strategy_num]
        ax.plot(result['aggr_episodes'], result['avg_rewards'], 
                label=f"Strategy {strategy_num}: {result['strategy_name']}", 
                color=colors[i], alpha=0.8, linewidth=2)
    
    ax.axvline(x=SATURATION_EPISODE, color='black', linestyle='--', alpha=0.5, label='Target Saturation')
    ax.axvline(x=WARMUP_EPISODES, color='grey', linestyle=':', alpha=0.5, label='Warm-up End')
    
    ax.set_title('Average Reward Comparison: All Strategies (500-episode windows) - Q-Learning CartPole')
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
    print("  - 6 Q-table files (.npy)")
    print("  - 6 detailed reward CSVs (per episode)")
    print("  - 6 aggregated CSVs (500-episode windows)")
    print("  - 6 individual strategy plots")
    print("  - 2 comparison plots (epsilon & rewards)")
    print(f"\n{'='*60}\n")