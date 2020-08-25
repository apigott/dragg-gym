import dragg
from dragg.reformat import Reformat
import gym
import dragg_gym

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2
#
# env = gym.make('dragg-v0')
# env.seed()
#
# model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./PPO2_5-houses/")
# model.learn(total_timesteps=1000, tb_log_name="first_run")
# print("trained")
#
# obs = env.reset()
# for _ in range(10):
#     action, _state = model.predict(obs)
#     obs, reward, done, info = env.step(action)
# env.agg.write_outputs(inc_rl_agents=False)

from tensorforce import Agent, Environment

# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='dragg-v0', max_episode_timesteps=500
)

# Instantiate a Tensorforce agent
agent = Agent.create(
    agent='tensorforce',
    environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
    memory=10000,
    update=dict(unit='timesteps', batch_size=64),
    optimizer=dict(type='adam', learning_rate=3e-4),
    policy=dict(network='auto'),
    objective='policy_gradient',
    reward_estimation=dict(horizon=20)
)

# Train for 300 episodes
for _ in range(300):

    # Initialize episode
    states = environment.reset()
    terminal = False

    while not terminal:
        # Episode timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

agent.close()
environment.close()
