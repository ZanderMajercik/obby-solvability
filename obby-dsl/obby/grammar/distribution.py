import random
import typing as tp
import numpy as np


class Distribution:
    def __init__(self, seed):
        self.seed = seed
        self.rng = random.Random()
        self.rng.seed(self.seed)
        self.min = -np.inf
        self.max = np.inf


class Constant(Distribution):
    def __init__(self, value: tp.Any, seed=0):
        super(Constant, self).__init__(seed)
        self.value = value
        self.min = value
        self.max = value

    def sample(self):
        return self.value
    
    def range(self, **kwargs):
        return [self.value]


class Binary(Distribution):
    def __init__(self, seed=None, ):
        super(Binary, self).__init__(seed)

    def sample(self):
        return self.rng.choices([0, 1])[0]
    
    def range(self, **kwargs):
        return [0, 1]


class UniformChoice(Distribution):
    def __init__(self, choices: list, weights: list[float] = None, seed=None):
        super(UniformChoice, self).__init__(seed)
        self.choices = choices
        self.weights = weights

    def sample(self):
        #TODO: restore
        if self.weights is None:
            return self.rng.choice(self.choices)
        else:
            return self.rng.choices(self.choices, weights=self.weights)[0]

    def range(self, **kwargs):
        return self.choices


class UniformInt(UniformChoice):
    def __init__(self, min: int, max: int, weights=None, seed=None):
        super(UniformInt, self).__init__(choices=range(min, max + 1), weights=weights, seed=seed)
        self.min = min
        self.max = max

    def sample(self) -> float:
        self.choices = range(self.min, self.max + 1)
        return super(UniformInt, self).sample()
    
    def range(self, **kwargs):
        return [i for i in range(self.min, self.max + 1)]


class Uniform(Distribution):
    def __init__(self, min: float, max: float, seed=None):
        super(Uniform, self).__init__(seed)
        self.min = min
        self.max = max

    def sample(self) -> float:
        return self.rng.random() * (self.max - self. min) + self.min
    
    def range(self, num=10):
        return np.linspace(self.min, self.max, num)
    

class Gaussian(Distribution):
    def __init__(self, mu: float, sigma: float, seed=None):
        super(Gaussian, self).__init__(seed)
        self.mu = mu
        self.sigma = sigma

    def sample(self) -> float:
        return self.rng.gauss(self.mu, self.sigma)
    
    def range(self, num=10, std=2):
        return np.linspace(self.mu - std * self.sigma, self.max + std * self.sigma, num)