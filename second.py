import os
import sys

if len(sys.argv) >= 2:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gym
import numpy as np
import tensorflow as tf
from keras import layers, Model
from keras.models import load_model

import matplotlib.pyplot as plt

from ddpg import OUActionNoise, DDPG
from rl.bank import Bank
from rl.data import Data

def learn():
    # env = gym.make("LunarLander-v2",continuous = True,render_mode="human",  enable_wind=False)
    env = gym.make("LunarLander-v2", continuous=True, enable_wind=False)


    def create_actor():
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=+0.003)

        inputs = layers.Input(shape=(env.observation_space.shape[0],))
        out = layers.Dense(256, activation='relu')(inputs)
        out = layers.Dense(256, activation='relu')(out)
        outputs = layers.Dense(2, activation='tanh', kernel_initializer=last_init)(out)

        # TODO MULTIPLY BY UPPER BOUND

        model = Model(inputs, outputs)
        return model


    def create_critic():
        state_input = layers.Input(shape=(env.observation_space.shape[0],))
        state_out = layers.Dense(16, activation='relu')(state_input)
        state_out = layers.Dense(32, activation='relu')(state_out)

        action_input = layers.Input(shape=(2,))
        action_out = layers.Dense(32, activation='relu')(action_input)

        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation='relu')(concat)
        out = layers.Dense(256, activation='relu')(out)
        outputs = layers.Dense(1)(out)

        model = Model([state_input, action_input], outputs)
        return model


    def policy(state, actor: Model, noise_object):
        sampled_actions = tf.squeeze(actor(state))
        noise = noise_object()

        sampled_actions = sampled_actions.numpy() + noise

        legal_action = np.clip(sampled_actions, -1., +1)

        return [np.squeeze(sampled_actions)]


    Data.MIN = env.observation_space.low
    Data.MAX = env.observation_space.high

    noise_obj = OUActionNoise(mean=np.zeros(1), std_deviation=float(DDPG.STD_DEV) * np.ones(1))

    observation, info = env.reset(seed=1, return_info=True)

    actor = create_actor()
    critic = create_critic()

    target_actor = create_actor()
    target_critic = create_critic()

    target_actor.set_weights(actor.get_weights())
    target_critic.set_weights(critic.get_weights())

    # Agent
    agent = DDPG(actor, critic, target_actor, target_critic)
    agent.bank = Bank()

    # History
    episode_reward_list = []
    avg_reward_list = []

    for ep in range(100000):
        observation, info = env.reset(return_info=True)
        episode_reward = 0

        last_state = None
        step = 0
        while True:
            step += 1
            print(f"step={step}", end='')
            action = policy(np.array([observation]), agent.actor, noise_obj)

            last_state = observation
            observation, reward, done, info = env.step(action[0])

            agent.bank.add(
                Data(np.array([last_state]),
                     action,
                     reward,
                     done,
                     np.array([observation]))
            )

            episode_reward += reward

            agent.learn()
            agent.update_target()

            if done:
                break
            print("\r", end='')

        episode_reward_list.append(episode_reward)
        avg_reward = np.mean(episode_reward_list[-40:])
        print(f"\nEP {ep}: {avg_reward}")
        avg_reward_list.append(avg_reward)

        if ep % 100 == 0:
            plt.plot(avg_reward_list)
            plt.savefig(f"saved-models/result-{ep}.png")
            plt.close()
            agent.critic.save(f"saved-models/model-critic-{ep}")
            agent.actor.save(f"saved-models/model-actor-{ep}")

    env.close()

def test(test_number):
    env = gym.make("LunarLander-v2",continuous=True, render_mode='human')

    Data.MIN = env.observation_space.low
    Data.MAX = env.observation_space.high

    def policy(state, actor: Model):
        sampled_actions = tf.squeeze(actor(state))
        # legal_action = np.clip(sampled_actions, -1., +1)
        return [np.squeeze(sampled_actions)]
        # return [np.array([0, 0.6])]

    actor = load_model(f"saved-models/model-actor-{test_number}")
    for ep in range(10000):
        observation, info = env.reset(return_info=True)

        while True:
            action = policy(np.array([observation]), actor)
            print(action)
            last_state = observation
            observation, reward, done, info = env.step(action[0])

            if done:
                break

    env.close()


if __name__ == "__main__":
    print(sys.argv)
    print(len(sys.argv))
    if len(sys.argv) >= 2:
        test(int(sys.argv[1]))
        exit()
    learn()


