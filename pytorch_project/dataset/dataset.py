import abc


class Dataset(abc.ABC):
    @abc.abstractmethod
    def to_cuda(self):
        pass

    @abc.abstractmethod
    def to_cpu(self):
        pass