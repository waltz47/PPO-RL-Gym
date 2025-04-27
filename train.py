import gymnasium as gym
import numpy as np
from ppo_agent import PPO
import torch
from collections import deque
from gymnasium.wrappers import RecordVideo
import os

def train():
    # Create video directory if it doesn't exist
    video_dir = "videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # Create environment
    env = gym.make('LunarLander-v3', render_mode="rgb_array")
    # Wrap environment with video recorder - record every 20 episodes
    env = RecordVideo(
        env, 
        video_dir,
        episode_trigger=lambda x: x % 20 == 0,  # Record every 20th episode
        name_prefix="lunarlander"
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize PPO agent
    agent = PPO(
        state_dim, 
        action_dim,
        epsilon_start=0.5,
        epsilon_end=0.01,
        epsilon_decay_episodes=300
    )
    
    # Training parameters
    max_episodes = 500  # Changed to 150 episodes
    max_steps = 1000
    batch_size = 64
    
    # Storage
    rewards_history = []
    average_rewards = deque(maxlen=100)
    
    # Training loop
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        next_states = []
        
        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob = agent.actor_critic.get_action(state_tensor)
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            states.append(state)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
            
            state = next_state
            episode_reward += reward
            
            if len(states) >= batch_size:
                agent.update(
                    np.array(states),
                    np.array(actions),
                    np.array(log_probs),
                    np.array(rewards),
                    np.array(dones),
                    np.array(next_states)
                )
                states, actions, log_probs, rewards, dones, next_states = [], [], [], [], [], []
            
            if done:
                break
        
        # Update with remaining steps
        if len(states) > 0:
            agent.update(
                np.array(states),
                np.array(actions),
                np.array(log_probs),
                np.array(rewards),
                np.array(dones),
                np.array(next_states)
            )
        
        # Update epsilon
        current_epsilon = agent.update_epsilon()
        
        rewards_history.append(episode_reward)
        average_rewards.append(episode_reward)
        avg_reward = np.mean(average_rewards)
        
        print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, Average Reward: {avg_reward:.2f}, Epsilon: {current_epsilon:.3f}")
    
    # Save the final model
    print("Training completed! Saving model...")
    agent.save('ppo_lunarlander.pth')
    env.close()

def record_video(model_path, num_episodes=5):
    """Record videos of a trained agent"""
    # Create video directory if it doesn't exist
    video_dir = "videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
        
    # Create environment with video recording
    env = gym.make('LunarLander-v3', render_mode="rgb_array")
    env = RecordVideo(
        env, 
        video_dir,
        episode_trigger=lambda x: True,  # Record every episode
        name_prefix="lunarlander_trained"
    )
    
    # Load trained agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, action_dim)
    agent.load(model_path)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, _ = agent.actor_critic.get_action(state_tensor)
            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            episode_reward += reward
            
        print(f"Test Episode {episode + 1}, Reward: {episode_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    train() 