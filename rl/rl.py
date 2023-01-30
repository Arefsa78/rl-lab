from random import randint, random
from rl.bank import Bank
from rl.dnn import Dnn
from rl.data import Data

import numpy as np
from keras.models import clone_model

class RL:
    GAMMA=0.9
    EPOCHS_PER_MEMORY=64
    LEARN_AFTER_N_STEP= 128
    BATCH_SIZE = 128
    SAVE_PER_STEP = 10000
    RANDOMNESS_PARAM = 0.4
    RANDOMNESS_UNTIL = 10000
    UPDATE_TARGET = 1000

    def __init__(self, load_dir:str="") -> None:
        if len(load_dir) > 0:
            self.load(load_dir)
            return
        self.online_network: Dnn = None
        self.target_network: Dnn = None
        self.bank: Bank = None

    def update_target(self):
        self.target_network = clone_model(self.online_network.model)
        self.target_network.set_weights(self.online_network.get_weights())
    
    
    def learn(self, batchs: int = 20, data: list[Data] = None):
        if len(self.bank.storage) < batchs:
            return

        data_batch: list[Data] = self.bank.get_batch(batchs)
        
        all_x = []
        all_y = []
        for data in data_batch:
            data_x = data.x().reshape((1,8))
            prediction = self.online_network.predict(data_x)[0]
            if data.is_done():
                prediction[data.action] = data.r()
            else:
                data_xp = data.xp().reshape((1,8))
                next_prediction = self.target_network.predict(data_xp, verbose=False)
                prediction[data.action] = RL.GAMMA * max(next_prediction[0]) + data.r()
            
            # prediction = prediction.reshape((1,4))
            all_x.append(data_x.reshape((8,1)))
            all_y.append(prediction)
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        self.online_network.learn(all_x, all_y, RL.EPOCHS_PER_MEMORY) 
    
    def action(self, state, randomness=False, step=0):
        normal_state = state - Data.MIN
        normal_state = normal_state / (Data.MAX - Data.MIN)

        if randomness and self.act_random(state, step):
            return self.action_random_index(state), True

        action = self.online_network.predict(state)
        return action, False
    
    def act_random(self, state, step):
        r = random()
        randomness = RL.RANDOMNESS_PARAM - step/RL.RANDOMNESS_UNTIL*RL.RANDOMNESS_PARAM
        if r < randomness:
            return True
        return False

    def action_random_index(self, state):
        action = randint(0, 3)
        actions = np.array([0. ,0. ,0. ,0. ])
        actions = actions.reshape(1,4)
        actions[0][action] = 1.
        return actions
        
    
    def save(self, step):
        self.online_network.save(f"saved-models/rl-model-{step}")

    def load(self, load_dir):
        self.online_network = Dnn(load_dir=load_dir)

    