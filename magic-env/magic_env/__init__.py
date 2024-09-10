from gymnasium.envs.registration import register

register(
    id="magic_env/CartPole-gravity",
    entry_point="magic_env.envs:GravityCartPoleEnv",
    vector_entry_point="magic_env.envs:GravityCartPoleVectorEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id="magic_env/CartPole-length",
    entry_point="magic_env.envs:LengthCartPoleEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id="magic_env/CartPole-both",
    entry_point="magic_env.envs:CartPoleEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id="magic_env/MountainCar",
    entry_point="magic_env.envs:MountainCarEnv",
    max_episode_steps=200,
    reward_threshold=-110.0,
)


register(
    id="magic_env/Acrobot",
    entry_point="magic_env.envs:AcrobotEnv",
    reward_threshold=-100.0,
    max_episode_steps=500,
)

register(
    id="magic_env/Pendulum",
    entry_point="magic_env.envs:PendulumEnv",
    max_episode_steps=200,
)

register(
    id="magic_env/Ant",
    entry_point="magic_env.envs:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="magic_env/Walker",
    entry_point="magic_env.envs:WalkerEnv",
    max_episode_steps=1000,
)

register(
    id="magic_env/HumanoidStandup",
    entry_point="magic_env.envs:MagicHumanoidStandupEnv",
    max_episode_steps=1000,
)
