import abc
from easydict import EasyDict


class ModelBase(object):
    def __init__(self, config):
        self._model = None
        self._config = EasyDict(config)

    @abc.abstractmethod
    def _build_model(self):
        pass

    def predict(self, x):
        return self._model.predict(x)

