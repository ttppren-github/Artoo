from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
import tensorflow as tf
import numpy as np


class DqnModel(Model):
    def __init__(self, state_dim, action_dim):
        super(DqnModel, self).__init__()

        self.__memory_size = 1024
        self.__memory = np.zeros((self.__memory_size, n_features * 2 + 2))
        self.__memory_counter = 0

        print(f'Create Dqn: \n'
              f'state dimension: {state_dim} \n'
              f'action dimension: {action_dim} \n')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dense1 = layers.Dense(120)
        self.q_value= layers.Dense(action_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        act = self.q_value(x)
        return act

    def record(self, cur_state, action, reward, new_state):
        transition = np.hstack((cur_state, [action, reward], new_state))
        index = self.__memory[self.__memory_counter, :] = transition
        self.__memory_counter = (self.__memory_counter + 1 ) % self.__memory_size


    def act(self, state):
        sample = random.random()
        # eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
        #     -step * self.epsilon_decay)
        if sample < 0.1:
            action = np.random.randint(0, self.n_actions)
        else:
            policy = self(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
            action = np.argmax(policy)

        return action

    def __build(self):
        pass
