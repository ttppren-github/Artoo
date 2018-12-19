import gym


class Env:
    TERMINAL = 'terminal'

    def __init__(self, model, train, config):
        self.__model = model
        self.__train = train
        self.__config = config
        self.__history = []

        print('Create environment: ' + config.name)
        self.__env = gym.make(config.name)

    def play(self):
        print('Train begin: count {}'.format(self.__config.episode_count))
        for e in range(1, self.__config.episode_count + 1):
            sts_cur = self.__env.reset()
            for i in range(self.__config.max_step):
                action = self.__model.predict(sts_cur)
                sts_next, reword, done, info = self.__env.step(action)

                if self.__config.render:
                    self.__env.render()

                if done:
                    self.__history.append((sts_cur, action, reword, None))
                    break
                else:
                    self.__history.append((sts_cur, action, reword, sts_next))
                self.__cur_sts = sts_next

            print(f'episode {e}: steps is {i}')

            if e % self.__config.train_every_episode == 0 and self.__train:
                self.__train.fit(self.__history)
        print('Train over')


    def close(self):
        self.__env.close()

    def sts_space(self):
        return self.__env.observation_space.shape[0]

    def act_space(self):
        return self.__env.action_space.n

    def get_history(self):
        return self.__history
