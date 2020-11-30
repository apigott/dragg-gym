import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from dragg.aggregator import Aggregator
from dragg.logger import Logger

import numpy as np
import itertools as it
import random
import pandas as pd

class MyAggregator(Aggregator):
    def __init__(self):
        super(MyAggregator, self).__init__()
        self.flush_redis()
        self.get_homes()

        self.max_rp = self.config['agg']['rl']['max_rp']
        self.all_rewards = []
        self.avg_load = 0.5 * self.max_poss_load
        self.agg_setpoint = 0
        self.max_load = 0

    def my_summary(self):
        self.collected_data["Summary"]["rl_rewards"] = self.all_rewards

class DRAGGEnv(gym.Env):
    def __init__(self, n_normalization_steps=10, verbose=False):
        super(DRAGGEnv, self).__init__()

        self.log = Logger("DRAGGEnv")
        self.verbose = verbose
        self.agg = MyAggregator()

        self.curr_episode = -1
        self.curr_step = -1
        self.action_episode_memory = []
        self.prev_action = 0
        self.avg_prev_action = 0

        # note that the action and state space are clipped to these values.
        action_low = np.array([-1], dtype=np.float32)
        action_high = np.array([1], dtype=np.float32)
        self.action_space = spaces.Box(action_low, action_high)

        obs_low = -np.ones(15, dtype=np.float32)
        obs_high = -np.ones(15, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high)
        # self.get_state()

        # note that the OpenAI gym and/or the Stable Baselines implementation of SAC
        # does NOT clip the reward value. It is recommended to normalize the reward value
        # to interface with the SAC algorithm.
        self.n_min_reward = -0.5
        self.n_max_reward = 0.5
        self.n_avg_reward = 0
        self.n_normalization_steps = n_normalization_steps
        self.normalize_reward_values()
        self.reset()

    def my_reward_func(self):
        frac = self.agg.agg_load / self.agg.max_poss_load
        reward = -(frac)**2 + frac
        return reward

    def get_reward(self):
        reward = self.my_reward_func()
        self.log.logger.debug(f"Reward BEFORE normalization {reward}.")
        reward = (reward - self.n_avg_reward) / (self.n_max_reward - self.n_min_reward)
        self.log.logger.debug(f"Reward AFTER normalization {reward}.")
        return reward

    def normalize_reward_values(self):
        min_reward = float("inf")
        max_reward = -float("inf")

        self.reset() # start at curr_episode = 0
        self.agg.case = "baseline" # let any output be written to the do nothing case

        for _ in range(self.n_normalization_steps):
            action = 0
            obs, reward, done, info = self.step(action)

            if reward < min_reward:
                min_reward = reward
            if reward > max_reward:
                max_reward = reward
            if len(self.agg.all_rewards) > 0:
                avg_reward = np.average(self.agg.all_rewards)

        self.n_max_reward = max_reward
        self.n_min_reward = min_reward
        self.n_avg_reward = avg_reward
        self.log.logger.info(f"Normalizing reward values against max reward: {self.n_max_reward}, min reward: {self.n_min_reward}, avg reward: {self.n_avg_reward}.")

    def take_action(self, action):
        self.action_episode_memory[self.curr_episode].append(action)
        self.reward_price = action
        self.prev_action = self.agg.reward_price[0] / self.agg.max_rp
        self.agg.avg_load += 0.1 * (self.agg.agg_load - self.agg.avg_load)
        self.avg_prev_action += 0.25 * (self.prev_action - self.avg_prev_action)
        self.agg.reward_price[0] = self.agg.max_rp * self.reward_price
        self.agg.redis_set_current_values()
        self.agg.run_iteration()
        self.agg.collect_data()

    def observe(self):
        return {"Current agg. load (kW)":                       {"value":self.agg.agg_load,                             "max":self.agg.max_poss_load,  "min":0,   "cyclical":False},
                "Predicted agg. load (kW)":                     {"value":self.agg.forecast_load,                        "max":self.agg.max_poss_load,  "min":0,   "cyclical":False},
                "Hour of day":                                  {"value":self.agg.timestep / self.agg.dt,               "max":24,                      "min":0,   "cyclical":True},
                "Week of year":                                 {"value":self.agg.timestep / (24 * self.agg.dt) / 7,    "max":52,                      "min":0,   "cyclical":True},
                "Avg. agg load for last {n} timesteps":         {"value":self.agg.avg_load,                             "max":self.agg.max_poss_load,  "min":0,   "cyclical":False},
                "Avg. agg load for the last {n} timesteps":     {"value":self.agg.agg_setpoint,                         "max":self.agg.max_poss_load,  "min":0,   "cyclical":False},
                "Change in OAT for next {n} timesteps":         {"value":self.agg.thermal_trend,                        "max":20,                      "min":0,   "cyclical":False},
                "OAT high for the day":                         {"value":self.agg.max_daily_temp,                       "max":25,                      "min":-5,  "cyclical":False},
                "OAT low for the day":                          {"value":self.agg.min_daily_temp,                       "max":10,                      "min":-5,  "cyclical":False},
                "Previous action":                              {"value":self.prev_action,                              "max":1,                       "min":-1,  "cyclical":False},
                "Avg. action for last {n} timesteps":           {"value":self.avg_prev_action,                          "max":1,                       "min":-1,  "cyclical":False},
                "GHI high for the day":                         {"value":self.agg.max_daily_ghi,                        "max":400,                     "min":0,   "cyclical":False},
                "Max load observed today":                      {"value":self.agg.max_load,                             "max":self.agg.max_poss_load,  "min":0,   "cyclical":False},
                }

    def get_state(self):
        state = self.observe()
        print(state)
        # if self.curr_step == -1:
        #     obs_low = -np.ones(len(state) + sum(state[k]['cyclical']==True for k in state), dtype=np.float32)
        #     obs_high = -np.ones(len(state) + sum(state[k]['cyclical']==True for k in state), dtype=np.float32)
        #     # obs_low = -np.ones(13, dtype=np.float32)
        #     # obs_high = -np.ones(13, dtype=np.float32)
        #     self.observation_space = spaces.Box(obs_low, obs_high)
        #     return

        # else:
        vals = []
        for k in state:
            if state[k]['cyclical'] == False:
                vals += [2 * state[k]['value'] / (state[k]['max'] - state[k]['min']) - 1]
            else:
                vals += [
                        np.sin(6.28 * state[k]['value'] / (state[k]['max'] - state[k]['min'])),
                        np.cos(6.28 * state[k]['value'] / (state[k]['max'] - state[k]['min']))
                ]

        if self.verbose:
            df = pd.DataFrame(state)
            print(df.T)
        # print("as a python list", vals)
        # print("as an np array", np.array(vals).shape)
        return np.array(vals, dtype=np.float32)

    def step(self, action): # done
        self.curr_step += 1
        self.take_action(action)
        obs = self.get_state()
        reward = self.get_reward()
        self.agg.prev_load = self.agg.agg_load
        self.agg.all_rewards += [reward]
        done = False
        return obs, reward, done, {}

    def reset(self):
        self.curr_step = -1
        self.curr_episode += 1
        self.action_episode_memory.append([])

        self.prev_action = 0
        self.prev_action_list = np.zeros(12)

        self.agg.setup_rl_agg_run()
        self.agg.reset_collected_data()
        self.agg.case = "rl_agg" # let the default be the rl_agg case
        self.agg.version = self.agg.config['simulation']['named_version']
        self.agg.set_run_dir()

        obs = self.get_state()
        return obs

    def _render(self, mode: str = "human", close: bool = False):
        return None

    def write_outputs(self):
        self.agg.write_outputs()

    def close(self):
        pass

    def seed(self, seed=12):
        random.seed(seed)
        np.random.seed(seed)
