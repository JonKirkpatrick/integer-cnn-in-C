from abc import ABC, abstractmethod

class IntOptimizer(ABC):
    def __init__(self, model):
        self.model = model
        self._register(model)

    @abstractmethod
    def _register(self, model):
        pass

    @abstractmethod
    def step(self, model):
        pass