import math
import numpy as np
import tensorflow as tf


class Noise:
    def __init__(self, task):
        self.size = np.prod(task.action_space.shape)
        self.low = task.action_space.low
        self.range = task.action_space.high - task.action_space.low

    def sample(self):
        return np.random.random((self.size)) * self.range / 10


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=None, theta=0.15, sigma=0.3):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self, mu=None):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = mu if mu is not None else self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class OUNoise2:
    def __init__(self, size, mu=None, theta=0.15, sigma=0.3, steps=100):
        self.n1 = OUNoise(size, mu, theta, sigma*math.sqrt(steps))
        self.n2 = OUNoise(size, mu, theta, sigma)
        self.steps = steps
        self.step = 0

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.n1.reset()
        self.n2.reset(self.n1.state)
        self.step = 0

    def sample(self):
        """Update internal state and return it as a noise sample."""
        if self.step % self.steps == 0:
            s1 = self.n1.sample()
            self.n2.reset(s1)
        else:
            s1 = self.n1.state

        s2 = self.n2.sample()

        return s1+s2



