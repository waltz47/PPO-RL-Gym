import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import imageio
import torch
import matplotlib
matplotlib.use('Agg') # Set backend before importing pyplot
import matplotlib.pyplot as plt

# Ensure the main directory exists
if not os.path.exists("humanoid"):
    os.makedirs("humanoid")

class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env, record_freq=30, plot_freq=20, video_dir="humanoid/videos", verbose=0): # Changed record_freq default and video_dir
        super().__init__(verbose)
        self.eval_env = eval_env
        self.record_freq = record_freq
        self.plot_freq = plot_freq
        self.video_dir = video_dir
        self.episode_count = 0
        self.episode_reward = 0
        self.rewards_history = []

        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self.episode_reward += reward

        done = self.locals["dones"][0]
        if done:
            self.rewards_history.append(self.episode_reward)
            # Only print episode number and final reward if verbose
            print(f"Episode: {self.episode_count}, Reward: {self.episode_reward:.2f}")
            self.episode_count += 1
            self.episode_reward = 0 # Reset for next episode

            # Record video every record_freq episodes
            if self.episode_count % self.record_freq == 0:
                if self.verbose > 0:
                    print(f"\nRecording video at episode {self.episode_count}...")
                self._record_video()
            
            # Plot rewards every plot_freq episodes
            if self.episode_count % self.plot_freq == 0:
                self._plot_rewards()

        return True

    def _record_video(self):
        env = self.eval_env
        obs, _ = env.reset()
        done = False
        truncated = False
        frames = []
        img = env.render()

        while img is not None and not (done or truncated):
            frames.append(img)
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
            img = env.render()

        # Save video using imageio
        if len(frames) > 0:
            video_path = os.path.join(self.video_dir, f"humanoid_episode_{self.episode_count}.mp4") # Changed filename
            imageio.mimsave(video_path, frames, fps=30)
            if self.verbose > 0:
                print(f"Saved video to {video_path}\n")

    def _plot_rewards(self):
        """Plots the rewards history and saves it to a file in the parent directory."""
        if len(self.rewards_history) > 0:
            plt.figure()  # Create a new figure
            plt.plot(self.rewards_history)
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title("Reward Over Time - Humanoid") # Changed title
            # Save the plot in the humanoid directory, not inside videos
            plot_path = os.path.join(os.path.dirname(self.video_dir), "reward_plot_humanoid.png") # Changed plot filename
            plt.savefig(plot_path)
            plt.close() # Close the figure to free memory

def main():
    # --- Parameters ---
    env_id = "Humanoid-v5" # Changed environment ID
    n_steps = 2048 # Steps collected before update (Consider increasing for Humanoid)
    batch_size = 128 # PPO Minibatch size (Consider increasing for Humanoid)
    total_timesteps = 10_000_000 # Increased timesteps for Humanoid
    record_freq_episodes = 50 # Record every N episodes (Set to 30 as requested)
    plot_freq_episodes = 20 # Plot every N episodes
    learning_rate = 3e-4 # Common LR for PPO on continuous tasks
    # -----------------

    # Create training environment
    print(f"Creating environment for {env_id}...")
    # Handle potential missing dependencies for Humanoid
    env = gym.make(env_id)

    # Create a separate env for evaluation/recording
    print("Creating evaluation environment...")
    eval_env = gym.make(env_id, render_mode="rgb_array")

    # --- Device Check ---
    # Force CPU usage
    device = "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu" # Original check
    print(f"Using device: {device}")
    # --------------------

    # Ensure Tensorboard log directory exists
    tensorboard_log_dir = "./humanoid/ppo_humanoid_tensorboard/" # Changed log directory
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)

    # Define network architecture (Consider larger network for Humanoid)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])) # Increased network size

    # Initialize PPO agent
    model = PPO(
        "MlpPolicy", # Use MlpPolicy for vector observations
        env,
        verbose=0,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0, # Often set to 0 for MuJoCo tasks
        vf_coef=0.5, # Default value
        max_grad_norm=0.5, # Default value
        tensorboard_log=tensorboard_log_dir,
        device=device,
        policy_kwargs=policy_kwargs # Add network architecture
    )

    # Create the callback
    callback = VideoRecorderCallback(
        eval_env=eval_env,
        record_freq=record_freq_episodes,
        plot_freq=plot_freq_episodes, # Add plot frequency
        video_dir="humanoid/videos", # Changed video directory
        verbose=0 # Set verbose to 0 to avoid double printing rewards
    )

    # Train the agent
    print(f"Starting training on {device}...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        # Save the model regardless of training completion status (if model exists)
        if 'model' in locals():
            # model_save_path = "humanoid/humanoid_ppo" # Changed save path
            # model.save(model_save_path)
            # print(f"Model saved to {model_save_path}")
            pass

        # Close the environments
        if 'env' in locals():
            env.close()
        if 'eval_env' in locals():
            eval_env.close()
        print("Training finished or interrupted. Environments closed.")

if __name__ == "__main__":
    main() 