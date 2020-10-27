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

run = ['rl', 'dn', 'tou']
mode = 'train' # or load

log = Logger("main")

env = gym.make('dragg-v0')
env._max_episode_steps = 1000

for l in [8, 12, 15, 20]:
    env.lam = l
    env.reset()

    model_name = f"lambda{l}"

    data_dir = os.path.expanduser(os.environ.get('DATA_DIR','data'))
    config_file = os.path.join(data_dir, os.environ.get('CONFIG_FILE', 'config.toml'))
    with open(config_file,'r') as f:
        data = toml.load(f)

    data['rl']['version'] = [model_name]
    with open(config_file,'w') as f:
        toml.dump(data, f)

    for _ in range(100):
        action = 0
        obs, reward, done, info = env.step(action)
    env.n_max_reward = env.max_reward
    env.n_min_reward = env.min_reward
    env.n_avg_reward = env.avg_reward
    log.logger.info(f"Normalizing the RL agent against",
                    f" Max Reward = {env.n_max_reward}",
                    f" Min Reward = {env.n_min_reward}",
                    f" Avg Reward = {env.n_avg_reward}")

    if 'rl' in run:
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
        data['rl']['version'] = ["dn-" + model_name]
        with open(config_file,'w') as f:
            toml.dump(data, f)

        obs = env.reset()
        for _ in range(240):
            action = 0
            obs, reward, done, info = env.step(action)

    if 'tou' in run:
        data['rl']['version'] = ["tou-" + model_name]
        data['rl']['utility']['base_price'] = 0.07
        data['rl']['utility']['tou_enabled'] = True
        with open(config_file,'w') as f:
            toml.dump(data, f)

        obs = env.reset()
        for _ in range(240):
            action = 0
            obs, reward, done, info = env.step(action)

        # reset to the non-TOU structure
        data['rl']['utility']['base_price'] = 0.10
        data['rl']['utility']['tou_enabled'] = False
        with open(config_file,'w') as f:
            toml.dump(data, f)
