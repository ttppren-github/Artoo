import abc
import numpy as np


class AgentBase(object):
    def __init__(self, state_dim, action_dim, e_greedy=0.9):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._epsilon = e_greedy

    @abc.abstractmethod
    def act(self, state):
        pass

    @abc.abstractmethod
    def save_model(self, m_path):
        pass

    @abc.abstractmethod
    def load_model(self, m_path):
        pass


class GymModel(object):
    def __init__(self, model, learning_rate=0.01, reward_decay=0.9, **kwargs):
        self.lr = learning_rate
        self.gamma = reward_decay
        self._model = model


    @abc.abstractmethod
    def learn(self, history):
        pass
