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

run = ['dn', 'rl']
mode = 'train' # or load

log = Logger("main")

env = gym.make('dragg-v0')
env._max_episode_steps = 1000

for l in [8]:
    env.lam = l

    model_name = f"lambda{str(l)}"
    log.logger.info(f"Model name set to: f{model_name}")

    env.agg.version = "dn-" + model_name

    env.reset()
    log.logger.info("Begining normalization process for reward function.")
    for _ in range(10):
        action = 0
        obs, reward, done, info = env.step(action)
    env.n_max_reward = env.max_reward
    env.n_min_reward = env.min_reward
    env.n_avg_reward = env.avg_reward
    log.logger.info([f"Normalizing the RL agent against: Max Reward = {str(env.n_max_reward)}, Min Reward = {str(env.n_min_reward)}, Avg Reward = {str(env.n_avg_reward)}"])

    if 'rl' in run:
        env.agg.version = model_name
        if mode == 'load':
            try:
                model = SAC.load(model_name)
                model.set_env(env)
            except:
                log.logger.warn(f"No model was found for version {model_name}.",
                f"Training a new model with name {model_name}.")

        elif mode == 'train':
            env.reset()
            model = SAC(LnMlpPolicy, env, learning_rate=0.03, verbose=1, tensorboard_log="tensorboard_logs")
            model.learn(total_timesteps=5000, tb_log_name=model_name)
            model.save(model_name)

        obs = env.reset()
        for _ in range(240):
            action, _state = model.predict(obs)
            obs, reward, done, info = env.step(action)

    if 'dn' in run:
        env.agg.version = "dn-" + model_name

        obs = env.reset()
        for _ in range(240):
            action = 0
            obs, reward, done, info = env.step(action)

    # if 'tou' in run:
    #     env.agg.version = "tou-" + model_name
    #     data['rl']['utility']['base_price'] = 0.07
    #     data['rl']['utility']['tou_enabled'] = True
    #     with open(config_file,'w') as f:
    #         toml.dump(data, f)
    #
    #     obs = env.reset()
    #     for _ in range(240):
    #         action = 0
    #         obs, reward, done, info = env.step(action)
    #
    #     # reset to the non-TOU structure
    #     data['rl']['utility']['base_price'] = 0.10
    #     data['rl']['utility']['tou_enabled'] = False
    #     with open(config_file,'w') as f:
    #         toml.dump(data, f)
