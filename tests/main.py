import dragg
from dragg.reformat import Reformat
import gym
import dragg_gym
from dragg.aggregator import Aggregator

from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C, SAC

env = gym.make('dragg-v0')
env.seed()

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=".tensorboard_logs")
model.learn(total_timesteps=5000, tb_log_name="random_agent")
model.save('ppo2_dragg_60-15min_AVGkW')
# # model = PPO2.load('ppo2_dragg_15min_3kW')

obs = env.reset()
for _ in range(10):
    # action, _state = model.predict(obs)
    action = 0
    obs, reward, done, info = env.step(action)
    print(env.agg.agg_load, env.agg.agg_setpoint)
env.agg.write_outputs(inc_rl_agents=False)

# r = Reformat()
# r.tf_main()
