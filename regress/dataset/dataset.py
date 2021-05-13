from torch.autograd import Variable

from machine import TrainConfig


class DataSet:
    _train_rate: float
    _test_rate: float
    _size: int
    _x: Variable
    _y: Variable
    _x_train: Variable
    _y_train: Variable
    _x_test: Variable
    _y_test: Variable

    def __init__(self,x:Variable,y:Variable,config:TrainConfig):
        self._x = x
        self._y = y
        self._train_rate = config.train_rate
        self._test_rate = config.test_rate
        train_size: int = int(self.size*self._train_rate)
        test_size: int = int(self.size*self._test_rate)
        self._x_train = self._x[:train_size]
        self._y_train = self._y[:train_size]

    @property
    def size(self):
        return len(self._x)

    @property
    def x_train(self):
        return self._x_train

    @property
    def y_train(self):
        return self._y_train

