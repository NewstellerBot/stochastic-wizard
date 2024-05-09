from gymnasium.envs.registration import register

register(
    id="MagicCartPole",
    entry_point="magic_cartpole.envs:MagicCartPoleEnv",
)
