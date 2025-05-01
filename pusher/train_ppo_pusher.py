import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import imageio
import torch

# Ensure the main directory exists
if not os.path.exists("pusher"):
    os.makedirs("pusher")

class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env, record_freq=20, video_dir="pusher/videos", verbose=0): # Updated video_dir
        super().__init__(verbose)
        self.eval_env = eval_env
        self.record_freq = record_freq
        self.video_dir = video_dir
        self.episode_count = 0
        self.episode_reward = 0
        self.episode_rewards = []

        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self.episode_reward += reward

        done = self.locals["dones"][0]
        if done:
            self.episode_rewards.append(self.episode_reward)
            # Only print episode number and final reward
            print(f"Episode: {self.episode_count}, Reward: {self.episode_reward:.2f}")
            self.episode_count += 1
            self.episode_reward = 0 # Reset for next episode

            # Record video every record_freq episodes
            if self.episode_count % self.record_freq == 0:
                print(f"\nRecording video at episode {self.episode_count}...")
                self._record_video()

        return True

    def _record_video(self):
        env = self.eval_env
        obs, _ = env.reset()
        done = False
        truncated = False # Pusher truncates after 100 steps
        frames = []
        img = env.render()

        while img is not None and not (done or truncated):
            frames.append(img)
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
            img = env.render()

        # Save video using imageio
        if len(frames) > 0:
            # Updated video filename
            video_path = os.path.join(self.video_dir, f"pusher_episode_{self.episode_count}.mp4")
            imageio.mimsave(video_path, frames, fps=30) # Default fps=30
            print(f"Saved video to {video_path}\n")


def main():
    # --- Parameters ---
    env_id = "Pusher-v5" # Updated environment ID
    n_steps = 2048 # Steps collected before update
    batch_size = 128 # PPO Minibatch size
    total_timesteps = 5_000_000 # Adjusted total timesteps for Pusher (shorter episodes)
    record_freq_episodes = 50 # Record every N episodes (as requested)
    learning_rate = 1e-4 # Common LR for PPO on continuous tasks
    # -----------------

    # --- Device Check ---
    # Force CPU usage (adjust if GPU is preferred and available)
    device = "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu" # Original check
    print(f"Using device: {device}")
    # --------------------

    # Create training environment
    # Pusher requires mujoco installation: pip install gymnasium[mujoco]
    print(f"Creating environment for {env_id}...")
    try:
        env = gym.make(env_id, max_episode_steps=150) # Add max_episode_steps
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Please ensure you have Mujoco installed: pip install gymnasium[mujoco]")
        return

    # Create a separate env for evaluation/recording
    print("Creating evaluation environment...")
    try:
        # Pusher uses rgb_array render mode implicitly when needed for MuJoCo >= 2.3.3
        eval_env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=150) # Add max_episode_steps
    except Exception as e:
        print(f"Error creating evaluation environment: {e}")
        return


    # Ensure Tensorboard log directory exists
    # Updated tensorboard log dir
    tensorboard_log_dir = "./pusher/ppo_pusher_tensorboard/"
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)

    # Define network architecture - Larger network for potentially complex dynamics
    policy_kwargs = dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])) # Made network deeper (3 layers)

    # Initialize PPO agent
    model = PPO(
        "MlpPolicy", # Use MlpPolicy for vector observations (Box space)
        env,
        verbose=0,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99, # Default discount factor
        gae_lambda=0.95, # Default GAE lambda
        clip_range=0.2, # Default clip range
        ent_coef=0.01, # Default entropy coefficient (can tune if needed)
        vf_coef=0.5, # Default value function coefficient
        max_grad_norm=0.5, # Default gradient norm clipping
        tensorboard_log=tensorboard_log_dir,
        device=device,
        policy_kwargs=policy_kwargs # Add network architecture
    )

    # Create the callback
    callback = VideoRecorderCallback(
        eval_env=eval_env,
        record_freq=record_freq_episodes,
        video_dir="pusher/videos" # Updated video dir
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
        # Consider saving the model even if training is interrupted
        model_save_path = "pusher/pusher_ppo_interrupted"
        model.save(model_save_path)
        print(f"Interrupted model saved to {model_save_path}")
    else:
        # Save the final model if training completes successfully
        model_save_path = "pusher/pusher_ppo_final" # Updated model save path
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")


    # Close the environments
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main() 