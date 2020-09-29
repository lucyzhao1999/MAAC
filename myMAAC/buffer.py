import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, size):
        self.storage = []
        self.size = int(size)
        self.nextIndex = 0

    def length(self):
        return len(self.storage)

    def clear(self):
        self.storage = []
        self.nextIndex = 0

    def add(self, obs, action, reward, nextObs, done):
        data = (obs, action, reward, nextObs, done)

        if self.nextIndex >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.nextIndex] = data
        self.nextIndex = (self.nextIndex + 1) % self.size

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]

        obsBatch, actionBatch, rewardsBatch, nextObsBatch, doneBatch = [], [], [], [], []
        for i in idxes:
            data = self.storage[i]
            obs, action, reward, nextObs, done = data
            obsBatch.append(np.array(obs, copy=False))
            actionBatch.append(np.array(action, copy=False))
            rewardsBatch.append(reward)
            nextObsBatch.append(np.array(nextObs, copy=False))
            doneBatch.append(done)
        return np.array(obsBatch), np.array(actionBatch), np.array(rewardsBatch), np.array(nextObsBatch), np.array(doneBatch)

