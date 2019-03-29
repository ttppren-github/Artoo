import numpy as np

import artoo


class Random(object):
    def __init__(self, state_dim, action_dim):
        self._action_dim = action_dim

    def act(self, states):
        r = np.random.randint(self._action_dim)
        return r


if __name__ == '__main__':
    agent = Random(4, 2)
    artoo.run(None, agent.act)
    print('bye bye')
