from stable_baselines3 import PPO
import gymnasium as gym
import magic_env

# from train.mujoco.ant import STEPS
from stable_baselines3.common.evaluation import evaluate_policy

STEPS = list(range(0, 10_000_000 + 1, 5_000))

# env = gym.make("InvertedDoublePendulum-v4", render_mode="rgb_array")
# magic_env = gym.make("magic_env/Ant", magic_mask=True)
# magic_agent = PPO("MlpPolicy", magic_env)

normal_env = gym.make("BipedalWalker-v3")

normal_env.model.body_mass *= 1e-16
normal_env.model.body_intertia *= 1e-16

normal_agent = PPO("MlpPolicy", normal_env)

steps_until_now = 0
for total_steps in STEPS:
    # magic_agent.learn(total_steps - steps_until_now)
    # print(
    #     f"[steps = {total_steps}]: magic = {evaluate_policy(magic_agent, magic_env, 20)}"
    # )
    print(
        f"[steps = {total_steps}]: normal = {evaluate_policy(normal_agent, normal_env, 20)}"
    )
    steps_until_now = total_steps

# vec_env = agent.get_env()
# obs = vec_env.reset()
# for i in range(2_000):
#     action, _state = agent.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")
