import gym
from rl.bank import Bank

from rl.data import Data
from rl.dnn import Dnn
from rl.history import History, RewardHistory
from rl.rl import RL
import time

now = time.time()


env = gym.make("LunarLander-v2", render_mode=None)

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

print(f"SETUP: {time.time() - now}s")

step = 0
while True:
    step += 1
    print(f"step={step}")

    now = time.time()
    # action = agent.action_index(observation)
    action = 1
    print(f"ACTION: {time.time() - now}s")


    last_state = observation

    now = time.time()
    observation, reward, done, info = env.step(action)
    print(f"STEP: {time.time() - now}s")

    now = time.time()
    # agent.bank.add(
    #     Data(last_state,
    #         action, 
    #         reward,
    #         done,
    #         observation)
    # )
    print(f"BANK: {time.time() - now}s")

    
    reward_history.update_step(reward)

    if step % RL.LEARN_AFTER_N_STEP == 0:
        now = time.time()
        agent.learn(RL.LEARN_AFTER_N_STEP) 
        print(f"LEARN: {time.time() - now}s")

    if done:
        now = time.time()
        observation, info = env.reset(return_info=True)
        reward_history.update_done()
        print(f"DONE: {time.time() - now}s")

    if step % History.SAVE_FIG_AFTER_STEPS == 0:
        now = time.time()
        reward_history.save_fig()    
        print(f"HISTORY: {time.time() - now}s")

    
    if step % RL.SAVE_PER_STEP == 0:
        now = time.time()
        agent.save(step)
        print(f"SAVE: {time.time() - now}s")
    
    




env.close()