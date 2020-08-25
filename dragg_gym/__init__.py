from gym.envs.registration import register

register(
    id='dragg-v0',
    entry_point='dragg_gym.envs:DRAGGEnv',
    max_episode_steps=1000,
    reward_threshold=0.0
)
