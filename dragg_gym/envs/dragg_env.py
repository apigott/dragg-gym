import gym
from gym import error, spaces, utils
from gym.utils import seeding
from dragg.aggregator import Aggregator
import numpy as np
import itertools as it
import random

class DRAGGEnv(gym.Env):
  # metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DRAGGEnv, self).__init__()
        self.agg = Aggregator()
        self.agg.case = 'rl_agg'

        self.agg.set_value_permutations()
        self.agg.set_dummy_rl_parameters()
        self.agg.flush_redis()
        self.agg.get_homes()

        self.curr_episode = -1
        self.curr_step = -1
        self.action_episode_memory = []

        self.MIN_PRICE = -0.2
        self.MAX_PRICE = 0.2
        # self.action_space = spaces.Discrete(41)
        self.action_space = spaces.Box(np.float32(np.array([-1])), np.float32(np.array([1])))
        obs_low = np.array([[0], # current load
                                        # [0], # forecasted load
                                        # [0], # time of day
                                        # [0] # predicted load over next X hours
                                        ])
        print(obs_low.shape)
        obs_high = np.array([[self.agg.config['community']['total_number_homes'][0] * 15], # current load
                                        # [self.agg.config['community']['total_number_homes'][0] * 15], # forecasted load
                                        # [24], # time of day
                                        # [self.agg.config['community']['total_number_homes'][0] * 100], # predicted load over next X hours
                                        ])
        self.observation_space = spaces.Box(obs_low, obs_high)

    def _get_reward(self, obs):
        reward = -1*(self.agg.agg_setpoint - self.agg.agg_load)**2
        return reward

    def _take_action(self, action):
        self.action_episode_memory[self.curr_episode].append(action)
        # self.reward_price = self.MIN_PRICE + ((self.MAX_PRICE - self.MIN_PRICE) / (self.action_space.n - 1) * action)
        self.reward_price = action
        self.agg.reward_price[0] = self.reward_price
        self.agg.redis_set_current_values()
        self.agg._run_iteration()
        self.agg.collect_data()

    def _get_state(self):
        self.reward_price[0] = 0
        forecast = self.agg._gen_forecast()
        # return (self.agg.agg_load, forecast[0], self.agg.timestep / self.agg.dt, sum(forecast))
        return self.agg.agg_load

    def step(self, action): # done
        self.curr_step += 1
        self._take_action(action)
        obs = self._get_state()
        reward = self._get_reward(obs)
        done = False
        return [[obs]], reward, done, {}

    def reset(self):
        self.curr_step = -1
        self.curr_episode += 1
        self.action_episode_memory.append([])

        self.agg.setup_rl_agg_run()
        self.agg.reset_baseline_data()
        self.agg.avg_load = 0

        obs = self._get_state()
        return obs

    def _render(self, mode: str = "human", close: bool = False) -> None:
        return None

    def close(self):
        pass

    def seed(self, seed=12):
        random.seed(seed)
        np.random.seed
