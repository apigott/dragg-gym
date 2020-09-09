import dragg
from dragg.reformat import Reformat
import gym
import dragg_gym
from dragg.aggregator import Aggregator

import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, ActorCriticPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C, SAC

class KerasPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(KerasPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=False)

        with tf.variable_scope("model", reuse=reuse):
            flat = tf.keras.layers.Flatten()(self.processed_obs)

            x = tf.keras.layers.Dense(1, activation="tanh", name='pi_fc_0')(flat)
            pi_latent = tf.keras.layers.Dense(1, activation="tanh", name='pi_fc_1')(x)

            x1 = tf.keras.layers.Dense(1, activation="tanh", name='vf_fc_0')(flat)
            vf_latent = tf.keras.layers.Dense(1, activation="tanh", name='vf_fc_1')(x1)

            value_fn = tf.keras.layers.Dense(1, name='vf')(vf_latent)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

model_name = 'a2c_discounted_learn'

# env = gym.make('dragg-v0')
env = DummyVecEnv([lambda: gym.make('dragg-v0')])
# env.seed()

# model = PPO2(MlpLnLstmPolicy, env, nminibatches=1, verbose=1, tensorboard_log=".tensorboard_logs")
model = A2C(MlpLnLstmPolicy, env, verbose=1, tensorboard_log=".tensorboard_logs")

model.learn(total_timesteps=5000, tb_log_name="random_agent")
model.save(model_name)
# model = PPO2.load(model_name)

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
    obs, reward , done, info = env.step(action)
# env.write_outputs(inc_rl_agents=False)

# r = Reformat()
# r.tf_main()
