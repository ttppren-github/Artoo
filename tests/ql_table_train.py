import pandas as pd
from base.train_base import TrainBase


class QLTableModelCoach(TrainBase):
    def __init__(self, model, config):
        super().__init__(model, config)

    def check_state_exist(self, state):
        if state not in self._model.q_table.index:
            # append new state to q table
            self._model.q_table = self._model.q_table.append(
                pd.Series(
                    [0] * len(self._model.actions),
                    index=self._model.q_table.columns,
                    name=state,
                    )
            )

    def fit(self, history):
        s, a, r, s_ = history[0]
        c_s = str(s)
        n_s = str(s_)
        self.check_state_exist(c_s)
        self.check_state_exist(n_s)
        q_predict = self._model.q_table.loc[c_s, a]
        if n_s is not None:
            q_target = r + self._config.gamma * self._model.q_table.loc[n_s, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self._model.q_table.loc[c_s, a] += self._config.lr * (q_target - q_predict)  # update
