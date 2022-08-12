from rl.bank import Bank
from rl.dnn import Dnn
from rl.data import Data
import numpy as np


class RL:
    GAMMA=0.9
    EPOCHS_PER_MEMORY=10
    LEARN_AFTER_N_STEP= 100

    def __init__(self) -> None:
        self.dnn: Dnn = None
        self.bank: Bank = None

    def learn(self, batchs: int = 20, data: list[Data] = None):
        if len(self.bank.storage) < batchs:
            return
            
        data_batch: list[Data] = self.bank.get_batch(batchs)
        all_x = []
        all_y = []
        for data in data_batch:
            data_x = data.x().reshape((1,8))
            prediction = self.dnn.predict(data_x)[0]
            if data.is_done():
                prediction[data.action] = data.r()
            else:
                data_xp = data.xp().reshape((1,8))
                next_prediction = self.dnn.predict(data_xp)
                prediction[data.action] = RL.GAMMA * max(next_prediction[0]) + data.r()
            
            prediction = prediction.reshape((1,4))
            all_x.append(data_x)
            all_y.append(prediction)
        self.dnn.learn(data_x, prediction, RL.EPOCHS_PER_MEMORY) 
    
    def action(self, state):
        return self.dnn.predict(state)
    
    def action_index(self, state):
        action = self.dnn.predict(state)
        return np.where(action == np.amax(action))[1][0]
    