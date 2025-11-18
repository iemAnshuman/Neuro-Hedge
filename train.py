import numpy as np
import torch
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt

from hedging_env import HedgingEnv
from ddpg_agent import DDPGAgent, ReplayBuffer

def train_ddpg():
    
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
    
    # FIXED: Better hyperparameters
    agent = DDPGAgent(state_dim, action_dim, tau=0.001, gamma=0.99, lr_actor=1e-4, lr_critic=1e-3)
    
    buffer_size = 1000000
    batch_size = 128  # Reduced batch size for more frequent updates
    replay_buffer = ReplayBuffer(buffer_size, batch_size)
    
    # FIXED: More training episodes
    max_episodes = 5000
    max_steps = env.n_steps
    
    # FIXED: Decaying noise for exploration
    noise_start = 0.3
    noise_end = 0.05
    noise_decay = 0.9995
    noise_scale = noise_start
    
    rewards_deque = deque(maxlen=100)
    all_episode_rewards = []
    all_actor_losses = []
    all_critic_losses = []
    
    # Warmup phase - fill buffer with random actions
    print("--- Warming up replay buffer ---")
    warmup_episodes = 50
    for episode in range(warmup_episodes):
        obs, info = env.reset()
        for step in range(max_steps):
            action = np.random.uniform(0.0, 1.0, size=action_dim)
            next_obs, reward, terminated, truncated, info = env.step(action)
            replay_buffer.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            if terminated:
                break
    
    print(f"--- Replay buffer warmed up with {len(replay_buffer)} experiences ---")
    print("--- Starting Training ---")
    
    for episode in range(max_episodes):
        # Reset noise at start of each episode
        agent.noise.reset()
        
        obs, info = env.reset()
        episode_reward = 0
        episode_actor_loss = []
        episode_critic_loss = []
        
        for step in range(max_steps):
            # FIXED: Use OU noise for exploration
            action = agent.select_action(obs, add_noise=True)
            
            # Additional exploration noise that decays
            if np.random.rand() < 0.1:  # 10% random exploration
                noise = np.random.normal(0, noise_scale, size=action_dim)
                action = np.clip(action + noise, 0.0, 1.0)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            replay_buffer.add(obs, action, reward, next_obs, terminated)
            
            obs = next_obs
            episode_reward += reward
            
            # FIXED: Update more frequently
            if len(replay_buffer) >= batch_size:
                actor_loss, critic_loss = agent.update_parameters(replay_buffer)
                if actor_loss is not None:
                    episode_actor_loss.append(actor_loss)
                    episode_critic_loss.append(critic_loss)
                
            if terminated:
                break
        
        # Decay exploration noise
        noise_scale = max(noise_end, noise_scale * noise_decay)
        
        rewards_deque.append(episode_reward)
        all_episode_rewards.append(episode_reward)
        
        if len(episode_actor_loss) > 0:
            all_actor_losses.append(np.mean(episode_actor_loss))
            all_critic_losses.append(np.mean(episode_critic_loss))
        
        avg_reward = np.mean(rewards_deque)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode+1}/{max_episodes} | Avg. Reward: {avg_reward:.4f} | Noise: {noise_scale:.4f}")
            
        # Save checkpoint every 1000 episodes
        if (episode + 1) % 1000 == 0:
            agent.save(f"neuro_hedge_agent_checkpoint_{episode+1}")
            print(f"--- Checkpoint saved at episode {episode+1} ---")
            
    print("--- Training Complete ---")
    
    agent.save("neuro_hedge_agent")
    print("--- Final Agent Saved ---")
    
    # Plot training progress
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot rewards
    axes[0].plot(all_episode_rewards, label='Episode Reward', alpha=0.3)
    axes[0].plot(np.convolve(all_episode_rewards, np.ones(100)/100, mode='valid'), 
                 label='Moving Avg (100 episodes)', linewidth=2)
    axes[0].set_title('Episode Rewards during Training')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot losses
    if len(all_actor_losses) > 0:
        axes[1].plot(all_actor_losses, label='Actor Loss', alpha=0.5)
        axes[1].plot(all_critic_losses, label='Critic Loss', alpha=0.5)
        axes[1].set_title('Training Losses')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_rewards.png', dpi=300)
    plt.show()
    
    print("\n--- Training metrics saved to training_rewards.png ---")


if __name__ == '__main__':
    train_ddpg()