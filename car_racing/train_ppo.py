import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
# Removed GrayscaleObservation import
import os
import numpy as np
import imageio
import torch
# Removed Wrapper import
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage # Added imports

# Ensure the main directory exists
if not os.path.exists("car_racing"):
    os.makedirs("car_racing")

class VideoRecorderCallback(BaseCallback):
    # Modified init to accept env_id and n_stack
    def __init__(self, eval_env, n_stack=3, env_id="CarRacing-v3", record_freq=20, video_dir="car_racing/videos", verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env # Env for rendering (must have render_mode='rgb_array')
        self.n_stack = n_stack
        self.env_id = env_id
        self.record_freq = record_freq
        self.video_dir = video_dir
        # Reverted to single-env tracking
        self.episode_count = 0
        self.episode_reward = 0
        self.episode_rewards = [] # Keep track of rewards for averaging later if needed

        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)

    def _on_step(self) -> bool:
        # Reverted to single-env logic (based on the first env in the VecEnv)
        # This assumes the underlying VecEnv is DummyVecEnv or similar with n_envs=1
        # If using multiple parallel envs, this logic needs adjustment
        if self.locals["dones"][0]: # Check done flag for the first env
            reward = self.locals["rewards"][0] # Get reward for the first env
            # Note: For episodic tasks, reward accumulation should ideally happen within the episode.
            # The current logic accumulates reward across steps until done, which might be incorrect
            # if the VecEnv resets environments individually.
            # Let's refine reward tracking based on 'infos' if available.
            # Assuming 'infos' contains terminal_observation and episode reward info
            info = self.locals["infos"][0]
            # --- REMOVED TEMPORARY DEBUG PRINT ---
            # print(f"[DEBUG] Episode End Info: {info}") 
            # -----------------------------------
            if "episode" in info:
                self.episode_reward = info["episode"]["r"] # Use final episode reward from info
                self.episode_rewards.append(self.episode_reward)
                print(f"Episode: {self.episode_count}, Reward: {self.episode_reward:.2f}")
                self.episode_count += 1
                # Resetting self.episode_reward explicitly is not needed if we always read from info

                # Record video every record_freq episodes
                if self.episode_count % self.record_freq == 0:
                    print(f"\nRecording video at episode {self.episode_count}...")
                    self._record_video()
            # Handle potential reward accumulation if info doesn't have 'episode' (less reliable)
            # else:
            #     self.episode_reward += reward # Accumulate if no episode info

        return True


    # Rewritten _record_video to handle observation space mismatch
    def _record_video(self):
        # Create a temporary env for stepping and prediction, wrapped like the training env
        print("  Creating temporary env for recording steps...")
        temp_env_raw = gym.make(self.env_id, continuous=True) # Ensure params match training
        temp_vec_env = DummyVecEnv([lambda: temp_env_raw])
        temp_stacked_env = VecFrameStack(temp_vec_env, n_stack=self.n_stack)
        temp_final_env = VecTransposeImage(temp_stacked_env) # This env yields obs matching model input
        print("  Temporary env created.")

        # Use the passed eval_env *only* for rendering
        render_env = self.eval_env # This one MUST have render_mode='rgb_array'
        _ = render_env.reset() # Reset render_env state
        img = render_env.render() # Get initial frame

        # Reset the stepping env and get initial *wrapped* observation
        obs = temp_final_env.reset()
        done = False
        frames = []
        step_count = 0
        max_steps = 1000 # Limit episode length during recording

        print("  Starting recording loop...")
        while img is not None and not done and step_count < max_steps:
            frames.append(img)
            # Predict action using the model and the *wrapped* observation
            action, _ = self.model.predict(obs, deterministic=True)

            # Step the *temporary wrapped* environment
            obs, _, dones, _ = temp_final_env.step(action)
            done = dones[0] # Get done state from the stepping env

            # Step the *render* environment with the same action to get the visual frame
            # Need to pass the action correctly (e.g., action[0] if action is wrapped)
            _, _, render_done, render_truncated, _ = render_env.step(action[0])
            img = render_env.render() # Get next frame

            # Handle termination from render_env
            if img is None or render_done or render_truncated:
                break
            step_count += 1

        print(f"  Recording loop finished after {step_count} steps.")

        # Close the temporary environments
        temp_final_env.close()
        print("  Temporary env closed.")

        # Close the rendering window of the render_env after recording
        # Avoid closing the main eval_env here if it's used elsewhere,
        # but CarRacing might require explicit viewer closing.
        if hasattr(render_env, 'close_viewer'):
             print("  Closing render_env viewer.")
             render_env.close_viewer()
        # elif hasattr(render_env, 'close'): # Avoid closing the potentially shared eval_env fully
        #      render_env.close()

        # Save video
        if len(frames) > 0:
            video_path = os.path.join(self.video_dir, f"car_racing_episode_{self.episode_count}.mp4")
            print(f"  Saving video ({len(frames)} frames) to {video_path}...")
            imageio.mimsave(video_path, frames, fps=30)
            print(f"Saved video to {video_path}\n")
        else:
            print("  No frames recorded, video not saved.\n")


def main():
    # --- Parameters ---
    env_id = "CarRacing-v3"
    n_stack = 3 # Number of frames to stack
    n_steps = 2048 # Steps collected before update
    batch_size = 128 # PPO Minibatch size
    total_timesteps = 1_000_000 # Adjust as needed
    record_freq_episodes = 10 # Record every N episodes
    learning_rate = 1e-4
    # Removed no_positive_reward_termination_steps parameter
    # -----------------

    # Create single training environment (RGB, default max_steps=1000)
    print(f"Creating single RGB environment for {env_id}...")
    # Use a lambda to defer creation for DummyVecEnv
    # Wrap the base environment creation with Monitor
    train_env_lambda = lambda: Monitor(gym.make(env_id, continuous=True))
    # Removed OffTrackTerminationWrapper application
    # Removed Grayscale and OffTrack wrappers

    # Create a separate *single* env for evaluation/recording (RGB, default max_steps=1000)
    print("Creating RGB evaluation environment (for rendering)...")
    eval_env = gym.make(env_id, continuous=True, render_mode="rgb_array")
    # Removed OffTrackTerminationWrapper application
    # Removed Grayscale and OffTrack wrappers

    # Wrap the training environment
    print(f"Wrapping training environment with DummyVecEnv, VecFrameStack (n_stack={n_stack}), and VecTransposeImage...")
    env = DummyVecEnv([train_env_lambda]) # Vectorize the environment
    env = VecFrameStack(env, n_stack=n_stack) # Stack frames
    env = VecTransposeImage(env) # Transpose dimensions for CNN (N, H, W, C*stack) -> (N, C*stack, H, W)
    print(f"Wrapped Training Env Observation Space: {env.observation_space}")

    # --- Device Check ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # --------------------

    # Initialize PPO agent
    model = PPO(
        "CnnPolicy",
        env, # Use the stacked, vectorized, transposed training environment
        verbose=0,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./car_racing/ppo_car_racing_tensorboard/",
        device=device
    )

    # Create the callback, passing necessary info
    callback = VideoRecorderCallback(
        eval_env=eval_env, # Use the original single env for rendering
        n_stack=n_stack,
        env_id=env_id,
        record_freq=record_freq_episodes,
        video_dir="car_racing/videos"
    )

    # Train the agent
    print(f"Starting training on {device}...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    # Save the model
    model_save_path = "car_racing/car_racing_ppo"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Close the environments
    env.close() # Close the VecEnv (should close underlying env)
    eval_env.close() # Close the single eval env used for rendering
    # Removed closing logic for raw/intermediate envs

if __name__ == "__main__":
    main() 