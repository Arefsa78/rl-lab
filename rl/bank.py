from rl.data import Data
from random import sample

class Bank:
    CHECK_SHAPE = False
    SIZE_LIMIT = 10000
    
    def __init__(self) -> None:
        self.storage: list[Data] = []
    
    def add(self, data: Data):
        if Bank.CHECK_SHAPE:
            pass
        
        if len(self.storage) > Bank.SIZE_LIMIT:
            self.pop[0]
        self.storage.append(data)
    
    def get_batch(self, batches=1) -> list[Data]:
        return sample(self.storage, batches)

    def __len__(self):
        return len(self.storage)