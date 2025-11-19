import numpy as np
import torch
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from hedging_env import HedgingEnv
from ddpg_agent import DDPGAgent
from black_scholes import BlackScholesBaseline

def run_evaluation(env, agent, num_episodes):
    all_pnls = []
    all_costs = []
    all_trade_counts = []
    all_actions_history = []
    
    is_rl_agent = hasattr(agent, 'select_action')
    
    if not is_rl_agent:
        bs_bot = agent
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        info = {}
        trade_count = 0
        prev_position = env.num_shares
        actions_this_ep = []
        
        while not terminated:
            if is_rl_agent:
                raw_action = agent.select_action(obs, add_noise=False)
                action = np.array(raw_action, dtype=np.float32).flatten()
            else:
                # FIX: Handle Normalized Observation
                # obs[1] is normalized price (S / K), so we multiply by K to get actual price
                time_to_expiry = obs[0]
                stock_price = obs[1] * env.K 
                
                current_time = env.T - time_to_expiry
                action_delta = bs_bot.get_action(current_time, stock_price)
                action = np.array([action_delta], dtype=np.float32)

            if abs(action[0] - prev_position) > 0.001:
                trade_count += 1
            
            actions_this_ep.append(action[0])
            prev_position = action[0]
                
            obs, reward, terminated, truncated, info = env.step(action)
        
        final_pnl = info.get('terminal_pnl', 0.0) 
            
        all_pnls.append(final_pnl)
        all_costs.append(env.total_cost)
        all_trade_counts.append(trade_count)
        all_actions_history.append(actions_this_ep)
        
    agent_type = 'DDPG' if is_rl_agent else 'Black-Scholes'
    print(f"\n--- Agent Type: {agent_type} ---")
    print(f"    Avg. Trades Per Episode: {np.mean(all_trade_counts):.2f}")
    print(f"    Min Trades: {np.min(all_trade_counts)}, Max Trades: {np.max(all_trade_counts)}")
    
    if len(all_actions_history) > 0:
        sample_actions = all_actions_history[0][:10]
        print(f"    Sample actions (first 10 steps): {[f'{a:.3f}' for a in sample_actions]}")
    
    return all_pnls, all_costs, all_actions_history

def evaluate_agent():
    
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    mu = r 
    sigma = 0.2
    dt = 1/52 
    c_rate = 0.002 

    sim_params = {'S0': S0, 'mu': mu, 'sigma': sigma, 'T': T, 'dt': dt}
    opt_params = {'K': K}

    env = HedgingEnv(sim_params, opt_params, r, c_rate)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    rl_agent = DDPGAgent(state_dim, action_dim)
    try:
        rl_agent.load("neuro_hedge_agent_checkpoint_5000")
        print("--- Trained DDPG Agent Loaded (5000 episodes) ---")
    except FileNotFoundError:
        try:
            rl_agent.load("neuro_hedge_agent")
            print("--- Trained DDPG Agent Loaded (generic) ---")
        except FileNotFoundError:
            print("--- ERROR: Trained agent files not found! ---")
            print("--- Please run train.py first! ---")
            return

    bs_agent = BlackScholesBaseline(K, T, r, sigma)
    print("--- Black-Scholes Baseline Agent Created ---")

    num_eval_episodes = 1000
    
    print(f"\n--- Running {num_eval_episodes} simulations for DDPG agent... ---")
    rl_pnls, rl_costs, rl_actions = run_evaluation(env, rl_agent, num_eval_episodes)
    
    print(f"\n--- Running {num_eval_episodes} simulations for Baseline agent... ---")
    bs_pnls, bs_costs, bs_actions = run_evaluation(env, bs_agent, num_eval_episodes)
    
    print("\n--- Evaluation Complete. Generating results... ---")
    
    rl_pnl_mean = np.mean(rl_pnls)
    rl_pnl_std = np.std(rl_pnls)
    rl_cost_mean = np.mean(rl_costs)
    rl_pnl_median = np.median(rl_pnls)
    rl_pnl_95percentile = np.percentile(rl_pnls, 95)
    rl_pnl_5percentile = np.percentile(rl_pnls, 5)

    bs_pnl_mean = np.mean(bs_pnls)
    bs_pnl_std = np.std(bs_pnls)
    bs_cost_mean = np.mean(bs_costs)
    bs_pnl_median = np.median(bs_pnls)
    bs_pnl_95percentile = np.percentile(bs_pnls, 95)
    bs_pnl_5percentile = np.percentile(bs_pnls, 5)
    
    results = {
        "Strategy": ["Neuro-Hedge (DDPG)", "Black-Scholes (Baseline)"],
        "Avg. P&L": [rl_pnl_mean, bs_pnl_mean],
        "Median P&L": [rl_pnl_median, bs_pnl_median],
        "P&L Std Dev": [rl_pnl_std, bs_pnl_std],
        "P&L 5th %ile": [rl_pnl_5percentile, bs_pnl_5percentile],
        "P&L 95th %ile": [rl_pnl_95percentile, bs_pnl_95percentile],
        "Avg. Txn Cost": [rl_cost_mean, bs_cost_mean]
    }
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("                      HEDGING PERFORMANCE RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(rl_pnls, bins=50, alpha=0.6, label='Neuro-Hedge', color='blue', density=True)
    axes[0, 0].hist(bs_pnls, bins=50, alpha=0.6, label='Black-Scholes', color='red', density=True)
    axes[0, 0].axvline(rl_pnl_mean, color='blue', linestyle='--', linewidth=2, label=f'RL Mean: {rl_pnl_mean:.2f}')
    axes[0, 0].axvline(bs_pnl_mean, color='red', linestyle='--', linewidth=2, label=f'BS Mean: {bs_pnl_mean:.2f}')
    axes[0, 0].axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes[0, 0].set_title('Final P&L Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Final P&L')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    box_data = [rl_pnls, bs_pnls]
    bp = axes[0, 1].boxplot(box_data, labels=['Neuro-Hedge', 'Black-Scholes'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    for box in bp['boxes']:
        box.set_alpha(0.5)
    axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[0, 1].set_title('P&L Distribution Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Final P&L')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    axes[1, 0].plot(rl_actions[0], label='Neuro-Hedge', alpha=0.7, linewidth=2, color='blue')
    axes[1, 0].plot(bs_actions[0], label='Black-Scholes', alpha=0.7, linewidth=2, color='red')
    axes[1, 0].set_title('Sample Hedging Strategy (Episode 1)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Hedge Ratio (# of shares)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.05)
    
    axes[1, 1].bar(['Neuro-Hedge', 'Black-Scholes'], [rl_cost_mean, bs_cost_mean], 
                   color=['blue', 'red'], alpha=0.6)
    axes[1, 1].set_title('Average Transaction Costs', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Transaction Cost')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('pnl_distribution.png', dpi=300)
    plt.show()
    
    print("\n--- Detailed results saved to pnl_distribution.png ---")

if __name__ == '__main__':
    evaluate_agent()