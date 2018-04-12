import gym

class env(object):
    def __init__(self, name, enable_render=False):
        self.__env = gym.make(name)
        self.__enable_render = enable_render

        self.stat = None
        self.reward = None
        self.terminal = None
        self.state_dim = self.__env.observation_space.shape[0]
        self.action_dim = self.__env.action_space.n

    def reset(self):
        self.s_t = self.__env.reset()
        return self.stat

    def step(self, act):
        (self.stat, self.reward, self.terminal, _info) = self.__env.step(act)

        if self.__enable_render:
            self.__env.render()
