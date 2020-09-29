import os
import toml

import dragg
from dragg.reformat import Reformat
import gym
import dragg_gym
from dragg.aggregator import Aggregator

import tensorflow as tf
# from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, ActorCriticPolicy
from stable_baselines.sac.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C, SAC, HER

model_name = 'higher_alpha03'

env = gym.make('dragg-v0')
env._max_episode_steps = 1000

# model = SAC(LnMlpPolicy, env, learning_rate=0.2 , verbose=1, tensorboard_log="tensorboard_logs")
# model.learn(total_timesteps=5000, tb_log_name=model_name)
# model.save(model_name)
model = SAC.load(model_name)
model.set_env(env)

obs = env.reset()
for _ in range(240):
    action, _state = model.predict(obs)
    # action = 0
    obs, reward, done, info = env.step(action)
env.agg.write_outputs(inc_rl_agents=False)

# for i in range(5):
#     model.learn(total_timesteps=5000, tb_log_name=(model_name+str(i)))
#     model.save(model_name+str(i))
#     temp_name = model_name+str(i)
#
#     data_dir = os.path.expanduser(os.environ.get('DATA_DIR','data'))
#     config_file = os.path.join(data_dir, os.environ.get('CONFIG_FILE', 'config.toml'))
#     with open(config_file,'r') as f:
#         data = toml.load(f)
#
#     data['rl']['version'] = temp_name
#     with open(config_file,'w') as f:
#         toml.dump(data, f)
#
#     env = gym.make('dragg-v0')
#
#     SAC.load(temp_name)
#
#     obs = env.reset()
#     state = None
#     done = [False for _ in range(1)]
#     for _ in range(240):
#         action, state = model.predict(obs, state=state, mask=done)
#         # action = 0
#         obs, reward , done, info = env.step(action)
# env.write_outputs(inc_rl_agents=False)
