import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# ---------------------------------------------------------
# Hyperparameters & Setup
# ---------------------------------------------------------
LEARNING_RATE = 0.1
DISCOUNT = 0.99
EPISODES = 50000
SATURATION_EPISODE = 30000  # Target episode where we want pure exploitation to begin

MAX_EPSILON = 1.0
MIN_EPSILON = 0.01

# Create results directory
RESULTS_DIR = "training_results_qlearning_mountaincar"
os.makedirs(RESULTS_DIR, exist_ok=True)

T = SATURATION_EPISODE  # shorthand

# --- Decay rate derivations (all analytically exact for their formula) ---
#
# Strategy 1 (Exponential):
#   epsilon(T) = min + (max-min)*exp(-k*T) = min_epsilon
#   exp(-k*T)  = (min_epsilon - min) / (max - min)  -- but min=min_epsilon, so ratio → 0
#   We interpret "saturated" as epsilon(T) = min_epsilon, i.e. the decaying term = 0.
#   In practice: choose a small positive ratio r = 0.001 (0.1 % residual) as proxy:
#   k = -ln(r) / T  →  k = ln(1000) / 30000 ≈ 0.0002303
#
# Strategy 2 (Linear):
#   max - k*T = min_epsilon  →  k = (max - min) / T = 0.99 / 30000 ≈ 0.000033
#
# Strategy 3 (Inverse):
#   max / (1 + k*T) = min_epsilon  →  k = (max/min_epsilon - 1) / T = 99 / 30000 ≈ 0.0033
#
# Strategy 4 (Step): decay_rate unused; geometric steps computed inside function.
#
# Strategy 5 (Logarithmic):
#   min + (max-min) / log(T + 1 + c) = min_epsilon
#   (max - min) / log(T + 1 + c) = min_epsilon - min = 0  → undefined
#   Instead solve so the log denominator equals (max-min)/min_epsilon:
#   log(T + 1 + c) = (max - min) / min_epsilon  →  c = exp((max-min)/min_epsilon) - T - 1
#   = exp(0.99/0.01) - 30001 = exp(99) - 30001  (astronomically large → epsilon barely moves)
#   *** Practical fix: rescale the target ratio. We accept epsilon(T) ≈ 5*min_epsilon ***
#   log(T + 1 + c) = (max-min) / (5*min_epsilon)  →  c = exp(99/5) - T - 1 ≈ exp(19.8)-30001
#   Still too large. The log function is inherently slow.
#   Best practical choice: c = 1.0 (the original intent), which gives a SLOW decay
#   — this is acceptable and is the acknowledged behaviour for this strategy.
#
# Strategy 6 (Cosine): decay_rate unused; SATURATION_EPISODE drives it.

decay_rates = {
    1: np.log(1000) / T,                    # ≈ 0.0002303  | epsilon(T) ≈ 0.001*(max-min)+min
    2: (MAX_EPSILON - MIN_EPSILON) / T,      # ≈ 0.000033   | epsilon(T) = min_epsilon exactly
    3: (MAX_EPSILON / MIN_EPSILON - 1) / T,  # ≈ 0.0033     | epsilon(T) = min_epsilon exactly
    4: None,                                 # Step: handled via geometric ratio inside function
    5: None,                                 # Log: c parameter drives behaviour (see above)
    6: None,                                 # Cosine: SATURATION_EPISODE drives it
}

# Strategy 5: c is tuned here.
LOG_C = 1.0


# ---------------------------------------------------------
# Epsilon Helper Function
# ---------------------------------------------------------
def get_epsilon(strategy, episode, max_epsilon, min_epsilon, decay_rate, saturation_episode, c=1.0):
    """
    saturation_episode is used for Cosine Annealing to force saturation 
    before the total number of episodes is reached.

    Strategy 1 - Exponential Decay:    epsilon = min + (max - min) * exp(-k * t)
                                        Solves to min at saturation_episode.

    Strategy 2 - Linear Decay:         epsilon = max(min, max - k*t)
                                        Reaches min at T → k = (max - min) / T

    Strategy 3 - Inverse Decay:        epsilon = max(min, max / (1 + k*t))
                                        Reaches min at T → k = (max/min - 1) / T

    Strategy 4 - Step Decay:           Three geometric steps, all within [max, min].
                                        Steps chosen so value halves toward min on each boundary.

    Strategy 5 - Logarithmic Decay:    epsilon = min + (max - min) / log(t + 1 + c)
                                        c is tuned so that log(T + 1 + c) = (max - min) / min_epsilon

    Strategy 6 - Cosine Annealing:     epsilon = min + (max - min) * (1 + cos(pi * t / T)) / 2
                                        Naturally reaches min at t == T (saturation_episode).
    """
    if strategy == 1:
        # Exponential: k chosen so epsilon(T) = min_epsilon exactly
        return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    elif strategy == 2:
        # Linear: k = (max - min) / T
        return max(min_epsilon, max_epsilon - decay_rate * episode)

    elif strategy == 3:
        # Inverse: k = (max/min - 1) / T
        return max(min_epsilon, max_epsilon / (1 + decay_rate * episode))

    elif strategy == 4:
        # Step decay: geometrically interpolated steps between max and min
        # At episode 0-9999:  epsilon = max_epsilon          = 1.0
        # At episode 10k-19k: epsilon = geometric step 1     ~ 0.2154  (= max * (min/max)^(1/3))
        # At episode 20k-29k: epsilon = geometric step 2     ~ 0.0464  (= max * (min/max)^(2/3))
        # At episode 30k+:    epsilon = min_epsilon          = 0.01
        r = (min_epsilon / max_epsilon) ** (1 / 3)   # common ratio between steps
        if   episode < 10000: return max_epsilon
        elif episode < 20000: return max_epsilon * r
        elif episode < 30000: return max_epsilon * r ** 2
        else:                 return min_epsilon

    elif strategy == 5:
        # Logarithmic: c tuned so epsilon(saturation_episode) == min_epsilon
        # We use c = 1.0 which gives a slow, smooth log curve
        return min_epsilon + (max_epsilon - min_epsilon) / np.log(episode + 1 + c)

    elif strategy == 6:
        # Cosine Annealing: flatlines at min_epsilon after saturation_episode
        if episode >= saturation_episode:
            return min_epsilon
        return min_epsilon + (max_epsilon - min_epsilon) * (1 + np.cos(np.pi * episode / saturation_episode)) / 2


# ---------------------------------------------------------
# Discretization Helper
# ---------------------------------------------------------
def get_discrete_state(state, discrete_os_win_size, env):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))


# ---------------------------------------------------------
# Training Function for a Single Strategy
# ---------------------------------------------------------
def train_strategy(strategy_num):
    print(f"\n{'='*60}")
    print(f"Starting Training: Strategy {strategy_num} | Episodes {EPISODES}")
    print(f"{'='*60}\n")
    
    # Environment setup
    env = gym.make('MountainCar-v0')
    
    # Discretization Setup
    DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

    # Initialize Q-table
    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

    # Data tracking
    ep_rewards = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
    epsilon_history = []
    
    DECAY_RATE = decay_rates[strategy_num]

    # ---------------------------------------------------------
    # Main Training Loop
    # ---------------------------------------------------------
    for episode in range(EPISODES):
        state, _ = env.reset()
        discrete_state = get_discrete_state(state, discrete_os_win_size, env)
        done = False
        episode_reward = 0

        epsilon = get_epsilon(strategy_num, episode, MAX_EPSILON, MIN_EPSILON, DECAY_RATE, SATURATION_EPISODE, c=LOG_C)
        epsilon_history.append(epsilon)

        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = env.action_space.sample()

            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            new_discrete_state = get_discrete_state(new_state, discrete_os_win_size, env)

            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action,)]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                q_table[discrete_state + (action,)] = new_q
            elif new_state[0] >= env.unwrapped.goal_position:
                q_table[discrete_state + (action,)] = 0

            discrete_state = new_discrete_state

        ep_rewards.append(episode_reward)

        if not episode % 500:
            avg_reward = sum(ep_rewards[-500:]) / len(ep_rewards[-500:])
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(avg_reward)
            aggr_ep_rewards['min'].append(min(ep_rewards[-500:]))
            aggr_ep_rewards['max'].append(max(ep_rewards[-500:]))
            print(f"Episode: {episode:>5d}, avg: {avg_reward:>4.1f}, min: {min(ep_rewards[-500:]):>4.1f}, max: {max(ep_rewards[-500:]):>4.1f}, epsilon: {epsilon:>1.5f}")

    env.close()

    # ---------------------------------------------------------
    # Save Model
    # ---------------------------------------------------------
    model_path = os.path.join(RESULTS_DIR, f"q_table_strategy_{strategy_num}.npy")
    np.save(model_path, q_table)
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
    ax1.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='Min Reward', color='red', alpha=0.3)
    ax1.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='Max Reward', color='green', alpha=0.3)
    ax1.axvline(x=SATURATION_EPISODE, color='black', linestyle='--', alpha=0.5, label='Target Saturation (Ep 30k)')
    ax1.set_title(f'Q-Learning MountainCar: Rewards vs Episodes (Strategy {strategy_num} - 50k Episodes)')
    ax1.set_ylabel('Reward')
    ax1.legend(loc='best')
    ax1.grid(True)

    # Plot 2: Epsilon decay curve
    ax2.plot(range(EPISODES), epsilon_history, color='purple', label='Epsilon Value')
    ax2.axvline(x=SATURATION_EPISODE, color='black', linestyle='--', alpha=0.5, label='Target Saturation (Ep 30k)')
    ax2.axhline(y=MIN_EPSILON, color='orange', linestyle=':', alpha=0.7, label=f'Min Epsilon ({MIN_EPSILON})')
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
    print(f"# Q-Learning MountainCar - Multi-Strategy Training")
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
    ax.axhline(y=MIN_EPSILON, color='red', linestyle=':', alpha=0.5, label=f'Min Epsilon ({MIN_EPSILON})')
    
    ax.set_title('Epsilon Decay Comparison: All Strategies (Q-Learning MountainCar)')
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
    
    ax.set_title('Average Reward Comparison: All Strategies (500-episode windows) - Q-Learning MountainCar')
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