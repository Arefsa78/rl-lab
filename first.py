import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from time import sleep
import gym
from rl.bank import Bank

from rl.data import Data
from rl.dnn import Dnn
from rl.history import History, RewardHistory
from rl.rl import RL

import sys
import numpy as np

def config():
    pass

def learn(load_number=None):
    env = gym.make("LunarLander-v2",  enable_wind=False)

    Data.MIN = env.observation_space.low
    Data.MAX = env.observation_space.high


    observation, info = env.reset(seed=1, return_info=True)

    # Agent
    agent = RL()
    if load_number:
        agent.load(f"saved-models/rl-model-{load_number}/")
    else:
        agent.online_network = Dnn(8, 4, [32, 64, 32], ['relu', 'relu', 'relu', 'relu'],"rl")
        agent.online_network.compile()
        
    agent.update_target()
    agent.bank = Bank()

    # History
    reward_history = RewardHistory()

    step = load_number if load_number else 0
    while True:
        step += 1
        if step % 1 == 0:
            print(f"step={step}", end='')

        actions, is_random = agent.action(observation, True, step)
        print(f"actions: {actions}")
        
        action = np.where(actions == np.amax(actions))[1][0]
    
        last_state = observation
        observation, reward, done, info = env.step(action)

        agent.bank.add(
            Data(last_state,
                action, 
                reward,
                done,
                observation)
        )
        
        if not is_random:
            reward_history.update_step(reward, actions)

        if step % RL.LEARN_AFTER_N_STEP == 0:
            print("\tlearning...", end='') 
            agent.learn(RL.BATCH_SIZE)


        if done:
            observation, info = env.reset(return_info=True)
            reward_history.update_done()
        
        if step % History.SAVE_FIG_AFTER_STEPS == 0:
            reward_history.save_fig(step)    
        
        if step % RL.UPDATE_TARGET == 0:
            agent.update_target()
        
        if step % RL.SAVE_PER_STEP == 0:
            agent.save(step)

    env.close()

def test(test_number):
    agent = RL(f"saved-models/rl-model-{test_number}/")
    env = gym.make("LunarLander-v2", render_mode='human')

    Data.MIN = env.observation_space.low
    Data.MAX = env.observation_space.high


    observation, info = env.reset(seed=1, return_info=True)

    while True:
        actions, _ = agent.action(observation)
        print(f"actions: {actions}")
        action = np.where(actions == np.amax(actions))[1][0]

        print(f"action: {action}")

        observation, reward, done, info = env.step(action)
        print(f"R:{reward}")

        if done:
            observation, info = env.reset(return_info=True)
            sleep(2)
        
    env.close()

def random_test():
    env = gym.make("LunarLander-v2", render_mode='human')

    observation, info = env.reset(seed=1, return_info=True)

    step = 0
    while True:
        step += 1
        action = 0
        if step % 5 == 0:
            action = 2
        sleep(1)

        print(f"action: {action}")

        observation, reward, done, info = env.step(action)
        print(f"R:{reward}")

        if done:
            observation, info = env.reset(return_info=True)
        
    env.close()



