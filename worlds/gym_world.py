# -*- coding: utf-8 -*-
import gym


class Gym(object):
    def __init__(self, name):
        self.__env = gym.make(name)

        self.__enable_render = False 
        self.__stat = None
        self.__reward = None
        self.__terminal = None
    
    @property
    def state_dim(self):
        self.state_dim = self.__env.observation_space.shape[0]

    @property
    def action_dim(self):
        self.action_dim = self.__env.action_space.n

    @property
    def enable_render(self):
        return self.__enable_render

    @enable_render.setter
    def enable_render(self, enable_render):
        self.__enable_render = enable_render

    @property
    def stat(self):
        return self.__stat

    @property
    def reword(self):
        return self.__reward
    
    @property
    def terminal(self):
        return self.__terminal

    def reset(self):
        self.__s_t = self.__env.reset()

    def step(self, act):
        (self.__stat, self.__reward, self.__terminal, _info) = self.__env.step(act)

        if self.__enable_render:
            self.__env.render()