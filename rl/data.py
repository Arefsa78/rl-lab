import numpy as np


class Data:
    MAX = None
    MIN = None

    def __init__(self, state, action, reward, done, next_state):
        self.state = state
        self.reward = reward
        self.done = done
        self.action = action
        self.next_state = next_state

        self.normalize()

    def x(self):
        return self.state

    def r(self):
        return self.reward

    def is_done(self):
        return self.done

    def xp(self):
        return self.next_state

    def a(self):
        return self.action

    def normalize(self):
        self.state -= Data.MIN
        self.state /= Data.MAX - Data.MIN

    def __repr__(self) -> str:
        return f"state: {self.state}\n" \
            f"reward: {self.reward}\n" \
            f"done: {self.done}\n" \
            f"action: {self.action}\n" \
            f"next_state: {self.next_state}"
