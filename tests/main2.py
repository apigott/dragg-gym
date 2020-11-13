import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import toml

import dragg
import gym
import dragg_gym
from dragg.logger import Logger

import tensorflow as tf
from stable_baselines.sac.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C, SAC, HER

run = ['dn', 'tou', 'rl']
mode = 'load' # or load
num_steps = 720

log = Logger("main")

env = gym.make('dragg-v0')
env._max_episode_steps = 1000

for i in [1,2]:
    for t in [2]:
        env.agg.max_setpont = t
        for l in [20]:
            env.agg.lam = l
            env.agg.max_rp = 0.02

            model_name = "l2agg-plus20"
            log.logger.info(f"Model name set to: {model_name}")

            env.agg.version = "dn-" + model_name + "-" + str(i)

            env.reset()
            log.logger.info("Begining normalization process for reward function.")
            for _ in range(50):
                action = 0
                obs, reward, done, info = env.step(action)
            max_reward = env.max_reward
            min_reward = env.min_reward
            avg_reward = env.avg_reward
            log.logger.info([f"Normalizing the RL agent against: Max Reward = {str(max_reward)}, Min Reward = {str(min_reward)}, Avg Reward = {str(avg_reward)}"])
            env.agg.n_max_reward = max_reward
            env.agg.n_min_reward = min_reward
            env.agg.n_avg_reward = avg_reward

            if 'rl' in run:
                env.agg.version = model_name + "-" + str(i)
                env.agg.config['rl']['utility']['tou_enabled'] = False
                env.agg.config['rl']['utility']['base_price'] = 0.1
                env.agg._build_tou_price()
                env.agg.redis_add_all_data()
                for h in env.agg.all_homes_obj:
                    h.initialize_environmental_variables()

                env.agg.version = model_name
                if mode == 'load':
                    try:
                        model = SAC.load(model_name)
                        model.set_env(env)
                    except:
                        log.logger.warn(f"No model was found for version {model_name}.",
                        f"Training a new model with name {model_name}.")
                        mode = 'train'

                elif mode == 'train':
                    env.reset()
                    model = SAC(LnMlpPolicy, env, learning_rate=0.03, verbose=1, tensorboard_log="tensorboard_logs")
                    model.learn(total_timesteps=5000, tb_log_name=model_name)
                    model.save(model_name)

                obs = env.reset()
                for _ in range(num_steps):
                    action, _state = model.predict(obs)
                    obs, reward, done, info = env.step(action)

            if 'dn' in run:
                env.agg.version = "dn-" + model_name + "-" + str(i)
                env.agg.config['rl']['utility']['tou_enabled'] = False
                env.agg.config['rl']['utility']['base_price'] = 0.1
                env.agg._build_tou_price()
                env.agg.redis_add_all_data()
                for h in env.agg.all_homes_obj:
                    h.initialize_environmental_variables()

                env.agg.version = "dn-" + model_name

                obs = env.reset()
                for _ in range(num_steps):
                    action = 0
                    obs, reward, done, info = env.step(action)

            if 'tou' in run:
                env.agg.version = "tou-" + model_name + "-" + str(i)
                env.agg.config['rl']['utility']['tou_enabled'] = True
                env.agg.config['rl']['utility']['base_price'] = 0.07
                env.agg._build_tou_price()
                env.agg.redis_add_all_data()
                for h in env.agg.all_homes_obj:
                    h.initialize_environmental_variables()

                obs = env.reset()
                for _ in range(num_steps):
                    action = 0
                    obs, reward, done, info = env.step(action)

                env.agg.config['rl']['utility']['tou_enabled'] = False
                env.agg.config['rl']['utility']['base_price'] = 0.1
                env.agg._build_tou_price()
                env.agg.redis_add_all_data()
                for h in env.agg.all_homes_obj:
                    h.initialize_environmental_variables()
