import abc


class TrainBase(object):
    def __init__(self, model, config: dict):
        self._model = model
        self._config = config

    @abc.abstractmethod
    def fit(self, data):
        pass

