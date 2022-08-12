import gym
from rl.bank import Bank

from rl.data import Data
from rl.dnn import Dnn
from rl.history import History, RewardHistory
from rl.rl import RL



env = gym.make("LunarLander-v2")

Data.MIN = env.observation_space.low
Data.MAX = env.observation_space.high


observation, info = env.reset(seed=1, return_info=True)

# Agent
agent = RL()
agent.dnn = Dnn(8, 4, [64,32], ['relu', 'relu'],"rl")
agent.dnn.compile()
agent.bank = Bank()

# History
reward_history = RewardHistory()

step = 0
while True:
    step += 1

    action = agent.action_index(observation)

    last_state = observation
    observation, reward, done, info = env.step(action)

    agent.bank.add(
        Data(last_state,
            action, 
            reward,
            done,
            observation)
    )
    
    reward_history.update_step(reward)

    if step % RL.LEARN_AFTER_N_STEP == 0:
        agent.learn(RL.LEARN_AFTER_N_STEP) 


    if done:
        observation, info = env.reset(return_info=True)
        reward_history.update_done()
    
    if step % History.SAVE_FIG_AFTER_STEPS:
        reward_history.save_fig()    
    
    if step % 100 == 0:
        print('\r', end='')
        print(f"step={step}", end='')
    
    if step % RL.SAVE_PER_STEP == 0:
        agent.dnn.
    
    




env.close()