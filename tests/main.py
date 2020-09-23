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

model_name = '6hr_sac-10dayEp'

env = gym.make('dragg-v0')
print("default max env steps", env._max_episode_steps)
env._max_episode_steps = 240
print("new max env steps", env._max_episode_steps)
# env = DummyVecEnv([lambda: gym.make('dragg-v0')])
# model = PPO2(MlpLnLstmPolicy, env, nminibatches=1, verbose=1, tensorboard_log="tensorboard_logs")

model = SAC(LnMlpPolicy, env, verbose=1, tensorboard_log="tensorboard_logs")
# model_class = SAC
# goal_selection_strategy = 'future'
# model = HER('MlpLnLstmPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy, verbose=1)

model.learn(total_timesteps=5000, tb_log_name=model_name)
model.save(model_name)
# model = PPO2.load(model_name)
# model.set_env(env)
# for i in range(5):
#     model.learn(total_timesteps=5000, tb_log_name=(model_name+str(i)))
#     model.save(model_name+str(i))

# obs = env.reset()
# for _ in range(240):
#     action, _state = model.predict(obs)
#     obs, reward, done, info = env.step(action)
# env.agg.write_outputs(inc_rl_agents=False)

obs = env.reset()
state = None
done = [False for _ in range(1)]
for _ in range(240):
    action, state = model.predict(obs, state=state, mask=done)
    # action = 0
    obs, reward , done, info = env.step(action)
# env.write_outputs(inc_rl_agents=False)

# r = Reformat()
# r.tf_main()
