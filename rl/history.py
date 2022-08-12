import matplotlib.pyplot as plt

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
    
    def update_step(self, reward):
        self.current_reward += reward
    
    def update_done(self):
        self.all_data.append(self.current_reward)
        self.current_reward = 0
    
    def history(self):
        return self.all_data
