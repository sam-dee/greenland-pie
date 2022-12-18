from abc import abstractmethod, ABC

from numpy import sqrt, zeros
from numpy.random import normal

PHASES = ('train', 'validate')


class BaseLayer(ABC):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

        self.w = normal(loc=0.0, scale=sqrt(2 / inputs), size=(inputs, outputs))
        self.b = zeros((1, outputs))

    @abstractmethod
    def __call__(self, x, phase):
        return x

    @property
    def trainable(self):
        return False

    @abstractmethod
    def backward(self, dy):
        """
        :param dy: значение градиента пришедшего от следующего слоя
        :return: значение градиента этого слоя
        """
        pass

    def update_weights(self, update_func):
        """
        обновление обучаемых параметров, если они есть, иначе ничего
        :param update_func: функция обновления, указано в презентации
        """
        pass

    @abstractmethod
    def get_nrof_trainable_params(self):
        """
        вычисление количества обучаемых параметров
        :return: количество обучаемых параметров
        """
        return 0
