import os

import numpy as np
import pandas as pd
from base.model_base import ModelBase


class QLearningTableModel(ModelBase):
    def __init__(self, config):
        super().__init__(config)

        self._build_model()

    def _build_model(self):
        self.actions = list(range(self._config.action_dim))
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

        self._model = self

    def predict(self, observation):
        state = str(observation)
        # action selection
        try:
            if np.random.uniform() < self._config.epsilon:
                # choose best action
                state_action = self.q_table.loc[state, :]
                # some actions may have the same value, randomly choose on in these actions
                action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            else:
                # choose random action
                action = np.random.choice(self._config.action_dim)
        except KeyError:
            action = np.random.choice(self._config.action_dim)

        return action

    def save_model(self, m_path):
        if not os.path.exists(m_path):
            os.mkdir(m_path)

        file = os.path.join(m_path, QLearningTableModel.MODEL_FILE)
        self.q_table.to_hdf(file, 'w')
        print(f'Save model to {file}')

    def load_model(self, m_path):
        file = os.path.join(m_path, QLearningTableModel.MODEL_FILE)
        if not os.path.exists(file):
            raise RuntimeError(f'{file} not exists!')
        self.q_table.read_hdf(file)
