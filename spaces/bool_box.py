from gym.spaces import Box
import numpy as np

class BoolBox(Box):
    """Matrix of boolean values with defined shape"""

    def __init__(self, shape):
        self.shape = shape
        self.dtype = np.bool
        super(BoolBox, self).__init__(low=0, high=2, shape=self.shape, dtype=self.dtype)

    def sample(self):
        """ Return a random sample from the space"""
        return np.random.randint(low=0, high=2, size=self.shape, dtype=self.dtype)

    def null_value(self):
        """ Return the space null value"""
        null = np.zeros(self.shape, self.dtype)
        return null

    def contains(self, x):
        return x.shape == self.shape