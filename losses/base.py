from abc import ABC, abstractmethod

class IntLoss(ABC):
    @abstractmethod
    def compute(self, logits, targets):
        pass