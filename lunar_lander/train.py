import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg') # Set backend before importing pyplot
import matplotlib.pyplot as plt

class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env, record_freq=20, plot_freq=20, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.record_freq = record_freq
        self.plot_freq = plot_freq
        self.episode_count = 0
        self.episode_reward = 0
        self.rewards_history = []
        if not os.path.exists("videos"):
            os.makedirs("videos")

    def _on_step(self) -> bool:
        # Update episode reward
        self.episode_reward += self.locals["rewards"][0]
        
        # Check if episode is done
        done = self.locals["dones"][0]
        if done:
            if self.verbose > 0:
                print(f"Episode {self.episode_count} - Reward: {self.episode_reward:.2f}")
            self.rewards_history.append(self.episode_reward)
            self.episode_count += 1
            self.episode_reward = 0
            
            # Record video every record_freq episodes
            if self.episode_count % self.record_freq == 0:
                self._record_video()
            
            # Plot rewards every plot_freq episodes
            if self.episode_count % self.plot_freq == 0:
                self._plot_rewards()
        
        return True
    
    def _record_video(self):
        # Record video
        env = self.eval_env
        obs, _ = env.reset()
        done = False
        truncated = False
        frames = []
        
        while not (done or truncated):
            frame = env.render()
            frames.append(frame)
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
        
        # Save video using imageio with macro_block_size=1 to prevent resizing
        if len(frames) > 0:
            video_path = f"videos/lunar_lander_episode_{self.episode_count}.mp4"
            imageio.mimsave(video_path, frames, fps=30, macro_block_size=1)

    def _plot_rewards(self):
        """Plots the rewards history and saves it to a file."""
        if len(self.rewards_history) > 0:
            plt.figure()  # Create a new figure
            plt.plot(self.rewards_history)
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title("Reward Over Time")
            plt.savefig("reward_plot.png")
            plt.close() # Close the figure to free memory

def main():
    # Create environment
    env = gym.make("LunarLanderContinuous-v3")
    eval_env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")

    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    # Create the callback
    callback = VideoRecorderCallback(eval_env=eval_env, record_freq=30, plot_freq=20, verbose=0)

    # Train the agent
    model.learn(
        total_timesteps=1_000_000,
        callback=callback,
        progress_bar=True
    )

    # Save the model
    model.save("lunar_lander_ppo")

if __name__ == "__main__":
    main() 