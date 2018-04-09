import random


class ReplayBuffer:
    def __init__(self, max_size = 1024):
        self.data = []
        self.max_size = max_size
        self.circ_insert_idx = 0

    def add(self, data):
        if len(self.data) < self.max_size:
            self.data.append(data)
        else:
            self.data[self.circ_insert_idx] = data
            self.circ_insert_idx = (self.circ_insert_idx + 1) % self.max_size

    def sample(self, size=64):
        return random.sample(self.data, size)

    def __len__(self):
        return len(self.data)