import tensorflow as tf
import numpy as np
from keras import optimizers, Model

from rl.bank import Bank


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class DDPG:
    BATCH_SIZE = 64

    ACTOR_OPTIMIZER = optimizers.Adam
    CRITIC_OPTIMIZER = optimizers.Adam

    CRITIC_LR = 0.002
    ACTOR_LR = 0.001

    TAU = 0.005
    GAMMA = 0.99

    STD_DEV = 0.2

    def __init__(self, actor: Model, critic: Model, target_actor: Model, target_critic: Model):
        self.actor = actor
        self.critic = critic

        self.target_actor = target_actor
        self.target_critic = target_critic

        self.bank: Bank = Bank()

        self.actor_optimizer = DDPG.ACTOR_OPTIMIZER(DDPG.ACTOR_LR)
        self.critic_optimizer = DDPG.CRITIC_OPTIMIZER(DDPG.CRITIC_LR)

    @tf.function
    def learn(self):
        if len(self.bank) <= DDPG.BATCH_SIZE:
            return
        data_batch = self.bank.get_batch(DDPG.BATCH_SIZE)

        state_batch = np.array([data.state for data in data_batch])
        action_batch = np.array([data.action for data in data_batch])
        reward_batch = np.array([data.reward for data in data_batch])
        next_state_batch = np.array([data.next_state for data in data_batch])

        with tf.GradientTape() as tape:
            target_action = self.target_actor(next_state_batch, training=True)
            y = reward_batch + DDPG.GAMMA * self.target_critic([next_state_batch, target_action], training=True)

            critic_value = self.critic([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_value = self.critic([state_batch, actions], training=True)

            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

    @tf.function
    def update_target(self):
        for (a, b) in zip(self.target_critic.variables, self.critic.variables):
            a.assign(b * DDPG.TAU + a * (1 - DDPG.TAU))

        for (a, b) in zip(self.target_actor.variables, self.actor.variables):
            a.assign(b * DDPG.TAU + a * (1 - DDPG.TAU))








































