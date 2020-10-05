import gym
from gym import error, spaces, utils
from gym.utils import seeding
from dragg.aggregator import Aggregator
import numpy as np
import itertools as it
import random

class DRAGGEnv(gym.Env):
    def __init__(self):
        super(DRAGGEnv, self).__init__()
        self.track_reward = 0
        self.min_reward = 0
        self.max_reward = -1000
        self.timestep = 0
        self.agg = Aggregator()
        self.agg.case = 'rl_agg'
        self.agg.avg_load = 30 # initialize setpoint
        self.agg.all_rewards = []

        self.agg.set_value_permutations()
        self.agg.set_dummy_rl_parameters()
        self.agg.flush_redis()
        self.agg.get_homes()

        self.curr_episode = -1
        self.curr_step = -1
        self.action_episode_memory = []
        self.prev_action = 0
        self.prev_action_list = np.zeros(12)

        action_low = np.array([-1
                            ], dtype=np.float32)
        action_high = np.array([1
                            ], dtype=np.float32)
        self.action_space = spaces.Box(action_low, action_high)
        obs_low = np.array([-1, # current load
                            -1, # forecasted load
                            -1, # sin(time of day)
                            -1, # cos(time of day)
                            -1, # sin(week of year)
                            -1, # cos(week of year)
                            -1, # predicted load over next X hours
                            -1, # weather trend
                            -1, # max daily temp
                            -1, # min daily temp
                            -1, # previous action
                            -1, # rolling avg previous action
                            -1, # max daily GHI
                            ], dtype=np.float32)
        obs_high = np.array([1, # current load
                            1, # forecasted load
                            1, # sin(time of day)
                            1, # cos(time of day)
                            1, # sin(week of year)
                            1, # cos(week of year)
                            1, # precent max load for past x timesteps
                            1, # weather trend
                            1, # max daily temp
                            1, # min daily temp
                            1, # previous action
                            1, # rolling avg previous action
                            1, # max daily GHI
                            ], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high)

    def get_reward(self, obs):
        # self.agg.avg_load += 0.5*(self.agg.agg_load - self.agg.avg_load)
        # sp = np.clip(self.agg.agg_setpoint, 45, 60)
        # sp = 50
        sp = self.agg.agg_setpoint
        reward = -1*(sp - self.agg.agg_load)**2 - 15*np.clip((self.agg.max_load - 60),0,None) - 15*np.clip((45 - self.agg.min_load),0,None)
        reward = (reward + 160) / (1000)
        # reward = -1*(self.agg.agg_load)**2
        # reward = (reward + 3724) / (-735 + 7084)
        self.track_reward += reward
        if reward < self.min_reward:
            self.min_reward = reward
        if reward > self.max_reward:
            self.max_reward = reward
        self.timestep += 1
        avg_reward = self.track_reward / self.timestep
        # print("avg",avg_reward, "min", self.min_reward, "max", self.max_reward)
        self.agg.prev_load = self.agg.agg_load
        # reward = -1 * (self.agg.agg_load - self.agg.avg_load)**2
        # print(reward)
        return reward

    def take_action(self, action):
        self.action_episode_memory[self.curr_episode].append(action)
        self.reward_price = action
        self.prev_action = self.agg.reward_price[0] / 0.09
        self.prev_action_list[:-1] = self.prev_action_list[1:]
        self.prev_action_list[-1] = self.prev_action
        self.agg.reward_price[0] = 0.09 * self.reward_price
        self.agg.redis_set_current_values()
        self.agg.run_iteration()
        self.agg.collect_data()


    def get_state(self):
        return np.array([2*(self.agg.agg_load/self.agg.config['community']['total_number_homes'][0] * 15) -1,
                        2*(np.sum(self.agg.forecast_load)/self.agg.config['community']['total_number_homes'][0] * 15) -1,
                        np.sin(3.14*(self.agg.timestep / self.agg.dt)/12),
                        np.cos(3.14*(self.agg.timestep / self.agg.dt)/12),
                        np.sin(3.14*(self.agg.timestep / (24 * self.agg.dt) / 7)/26),
                        np.cos(3.14*(self.agg.timestep / (24 * self.agg.dt) / 7)/26),
                        2*(np.average(self.agg.tracked_loads[-4:]) / self.agg.max_load) -1,
                        2*(self.agg.thermal_trend - (-10))/20-1,
                        2*(self.agg.max_daily_temp - (-1))/23-1,
                        2*(self.agg.min_daily_temp - (-1))/23-1,
                        self.prev_action,
                        np.average(self.prev_action_list),
                        self.agg.max_daily_ghi /400 - 1])

    def step(self, action): # done
        self.curr_step += 1
        self.take_action(action)
        obs = self.get_state()
        reward = self.get_reward(obs)
        self.agg.all_rewards += [reward]
        self.agg.write_outputs(inc_rl_agents=False) # how to do this better for DummyVecEnv??
        done = False
        return obs, reward, done, {}

    def reset(self):
        self.curr_step = -1
        self.curr_episode += 1
        self.action_episode_memory.append([])

        self.agg.setup_rl_agg_run()
        self.agg.reset_baseline_data()
        self.agg.avg_load = 0

        obs = self.get_state()
        return obs

    def _render(self, mode: str = "human", close: bool = False):
        return None

    def write_outputs(self):
        self.agg.write_outputs(inc_rl_agents=False)

    def close(self):
        pass

    def seed(self, seed=12):
        random.seed(seed)
        np.random.seed
