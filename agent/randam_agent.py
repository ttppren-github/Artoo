import random

class RandomAgent(object):
    def __init__(self, state_dim, action_dim):
        self.__state_dim = state_dim
        self.__action_dim = action_dim 
    
    def act(self, states):
        r = random.randint(self.__action_dim)
        rv = [0] * self.__action_dim
        rv[r] = 1
        
        return rv
