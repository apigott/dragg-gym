import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import toml
import numpy as np

import dragg
import gym
import dragg_gym
from dragg.logger import Logger

import tensorflow as tf
from stable_baselines.sac.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C, SAC, HER

run = ['dn']
mode = 'load' # or load
num_steps = 720
checkpoint_interval = 24

log = Logger("main")

env = gym.make('dragg-v0')
env._max_episode_steps = 1000

env.agg.max_rp = 0.02

model_name = "l2agg30"
env.agg.version = model_name
log.logger.info(f"Model name set to: {model_name}")

if 'rl' in run:
    env.agg.config['agg']['tou_enabled'] = False
    env.agg.config['agg']['base_price'] = 0.1
    env.agg._build_tou_price()
    env.agg.redis_add_all_data()
    for h in env.agg.all_homes_obj:
        h.initialize_environmental_variables()

    if mode == 'load':
        try:
            model = SAC.load(model_name)
            model.set_env(env)
        except:
            log.logger.warning(f"No model was found for version {model_name}. Training a new model with name {model_name}.")
            mode = 'train'

    if mode == 'train':
        env.reset()
        env.agg.case = 'rl_agg'
        model = SAC(LnMlpPolicy, env, learning_rate=0.03, verbose=1, tensorboard_log="tensorboard_logs")
        # note that the env won't record MPCCalc output for the training period
        model.learn(total_timesteps=5000, tb_log_name=model_name)
        model.save(model_name)

    obs = env.reset()
    env.agg.case = 'rl_agg'
    for t in range(1, num_steps+1):
        action, _state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if (t % checkpoint_interval == 0) or (t == num_steps):
            env.agg.write_outputs()

if 'dn' in run:
    env.agg.config['agg']['tou_enabled'] = False
    env.agg.config['agg']['base_price'] = 0.1
    env.agg._build_tou_price()
    env.agg.redis_add_all_data()
    for h in env.agg.all_homes_obj:
        h.initialize_environmental_variables()

    obs = env.reset()
    env.agg.case = 'baseline'

    for t in range(1, num_steps+1):
        action = 0
        obs, reward, done, info = env.step(action)
        if (t % checkpoint_interval == 0) or (t == num_steps):
            env.agg.write_outputs()

if 'tou' in run:
    env.agg.config['agg']['tou_enabled'] = True
    env.agg.config['agg']['base_price'] = 0.07
    env.agg._build_tou_price()
    env.agg.redis_add_all_data()
    for h in env.agg.all_homes_obj:
        h.initialize_environmental_variables()

    obs = env.reset()
    env.agg.case = 'tou'

    for t in range(1, num_steps+1):
        action = 0
        obs, reward, done, info = env.step(action)
        if (t % checkpoint_interval == 0) or (t == num_steps):
            env.agg.write_outputs()
