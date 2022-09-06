from cProfile import label
import matplotlib.pyplot as plt
import numpy as np

class History:
    SAVE_FIG_AFTER_STEPS = 1000

    def __init__(self) -> None:
        self.all_data = []

    def update_step(self, reward):
        pass
    def update_done(self):
        pass
    
    def history(self):
        pass

    def save_fig(self):
        plt.plot(self.all_data)
        plt.savefig("result.png")
        plt.close()

class RewardHistory(History):
    def __init__(self) -> None:
        super().__init__()
        self.current_reward = 0
        self.actions_rewards = np.array([0.,0.,0.,0.])
        self.n_step = 0
    
    def update_step(self, reward, actions):
        self.current_reward += reward
        self.actions_rewards += actions[0]
        self.n_step += 1
    
    def update_done(self):
        self.all_data.append((self.current_reward / self.n_step/5, self.actions_rewards/self.n_step))
        self.current_reward = 0
    
    def history(self):
        return self.all_data
    
    def save_fig(self, step=None):
        reward = [x[0] for x in self.all_data]
        actions = [[x[1][k] for x in self.all_data] for k in range(4)]
        plt.plot(reward, 'black')
        plt.plot(actions[0], 'red', label=0)
        plt.plot(actions[1], 'blue', label=1)
        plt.plot(actions[2], 'green', label=2)
        plt.plot(actions[3], 'yellow', label=3)
        if step:
            plt.savefig(f"saved-models/result-{step}.png")
        else:
            plt.savefig("result.png")
        plt.close()
