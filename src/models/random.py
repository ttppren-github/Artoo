import numpy as np
from .agent_base import AgentBase


class Random(AgentBase):
    def __init__(self, state_dim, action_dim):
        super(Random, self).__init__(state_dim, action_dim)

    def act(self, states):
        r = np.random.randint(self._action_dim)
        return r
