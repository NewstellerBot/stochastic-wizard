from gymnasium.envs.registration import register

register(
    id="MagicCartPole",
    entry_point="magic_cartpole.envs:MagicCartPoleEnv",
    vector_entry_point="magic_cartpole.envs:MagicCartPoleVectorEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)
