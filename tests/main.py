import os
import toml

import dragg
import gym
import dragg_gym
from dragg.logger import Logger

import tensorflow as tf
# from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, ActorCriticPolicy
from stable_baselines.sac.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C, SAC, HER

log = Logger("main")

model_name = 'alpha03-plus-peak-valley'

data_dir = os.path.expanduser(os.environ.get('DATA_DIR','data'))
config_file = os.path.join(data_dir, os.environ.get('CONFIG_FILE', 'config.toml'))
with open(config_file,'r') as f:
    data = toml.load(f)

data['rl']['version'] = [model_name]
with open(config_file,'w') as f:
    toml.dump(data, f)

env = gym.make('dragg-v0')
env._max_episode_steps = 1000

model = SAC(LnMlpPolicy, env, learning_rate=0.03, verbose=1, tensorboard_log="tensorboard_logs")
model.learn(total_timesteps=5000, tb_log_name=model_name)
model.save(model_name)

# model = SAC.load(model_name)
# model.set_env(env)

obs = env.reset()
for _ in range(240):
    action, _state = model.predict(obs)
    # action = 0
    obs, reward, done, info = env.step(action)

# alphas = [3,5,7,9]
# for i in range(5):
# for a in alphas:
#     temp_name = model_name+"0"+str(a)
#
#     data_dir = os.path.expanduser(os.environ.get('DATA_DIR','data'))
#     config_file = os.path.join(data_dir, os.environ.get('CONFIG_FILE', 'config.toml'))
#     with open(config_file,'r') as f:
#         data = toml.load(f)
#
#     data['rl']['version'] = [temp_name]
#     with open(config_file,'w') as f:
#         toml.dump(data, f)
#     log.logger.info(f"rewrote config file for alpha {a}")
#
#     env = gym.make('dragg-v0')
#     model = SAC(LnMlpPolicy, env, learning_rate=(0.01 * a) , verbose=1, tensorboard_log="tensorboard_logs")
#     model.learn(total_timesteps=5000, tb_log_name=(temp_name))
#     model.save(temp_name)
#     log.logger.info(f"learned and saved model for alpha {a}")
#     # SAC.load(temp_name)
#
#     obs = env.reset()
#     for _ in range(240):
#         action, state = model.predict(obs)
#         obs, reward , done, info = env.step(action)
#     log.logger.info(f"completed 10 day simulation for alpha {a}")
