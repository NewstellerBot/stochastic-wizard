import gymnasium as gym
import magic_cartpole

from stable_baselines3 import PPO

env = gym.make("MagicCartPole/both", render_mode="rgb_array")
model = PPO.load("checkpoints/both/wizard/9/50000", env=env)

vec_env = model.get_env()
obs = vec_env.reset()

for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
