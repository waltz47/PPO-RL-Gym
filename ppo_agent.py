import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device)
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Move entire model to device
        self.to(device)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
        
    def forward(self, state):
        # Actor (using log softmax for numerical stability)
        action_logits = self.actor(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        # Critic
        value = self.critic(state)
        
        return action_probs, value
    
    def get_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.cpu(), action_log_prob.cpu()
    
    def evaluate_actions(self, state, action):
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return action_log_probs, value, entropy

class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=0.5,
        epsilon_end=0.01,
        epsilon_decay_episodes=200,
        epochs=10,
        entropy_coef=0.01,
        value_coef=0.5,
    ):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon = epsilon_start
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.current_episode = 0
        
    def update_epsilon(self):
        """Update epsilon using exponential decay"""
        self.current_episode += 1
        progress = min(1.0, self.current_episode / self.epsilon_decay_episodes)
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-5 * progress)
        return self.epsilon

    def get_epsilon(self):
        """Get current epsilon value"""
        return self.epsilon
        
    def update(self, states, actions, old_log_probs, rewards, dones, next_states):
        # Skip update if batch is empty
        if len(states) == 0:
            return
            
        # Convert to tensor and move to device
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        
        # Compute returns and advantages
        with torch.no_grad():
            _, next_values = self.actor_critic(next_states)
            _, values = self.actor_critic(states)
            
            next_values = next_values.squeeze(-1)
            values = values.squeeze(-1)
            
            advantages = torch.zeros_like(rewards).to(device)
            returns = torch.zeros_like(rewards).to(device)
            
            running_returns = 0 if dones[-1] else next_values[-1].item()
            running_advantage = 0
            gae_lambda = 0.95
            
            for t in reversed(range(len(rewards))):
                running_returns = rewards[t] + self.gamma * running_returns * (1 - dones[t])
                returns[t] = running_returns
                
                # GAE calculation
                delta = rewards[t] + self.gamma * next_values[t].item() * (1 - dones[t]) - values[t].item()
                running_advantage = delta + self.gamma * gae_lambda * running_advantage * (1 - dones[t])
                advantages[t] = running_advantage
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update with mini-batches
        batch_size = len(states)
        mini_batch_size = max(batch_size // 4, 1)  # Ensure mini_batch_size is at least 1
        
        for _ in range(self.epochs):
            indices = torch.randperm(batch_size).to(device)
            
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]
                
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                log_probs, current_values, entropy = self.actor_critic.evaluate_actions(mb_states, mb_actions)
                current_values = current_values.squeeze(-1)
                
                # Compute ratio and surrogate losses
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * mb_advantages
                
                # Value loss with clipping
                value_pred_clipped = values[mb_indices] + torch.clamp(
                    current_values - values[mb_indices],
                    -self.epsilon,
                    self.epsilon
                )
                value_losses = (current_values - mb_returns).pow(2)
                value_losses_clipped = (value_pred_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                
                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.value_coef * value_loss 
                    - self.entropy_coef * entropy
                )
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
            
    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)
        
    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=device)) 