import logging
import torch
import gymnasium as gym
import magic_cartpole

from stable_baselines3 import PPO

env = gym.make("MagicCartPole", render_mode="rgb_array")
device = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # mps seems to work slower than cpu

logging.info(f"Using {device}")

model = PPO("MlpPolicy", env, verbose=1, device=device)
model.learn(total_timesteps=500_000)

vec_env = model.get_env()
obs = vec_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
