from abc import ABC, abstractmethod

class BaseClient(ABC):
    @abstractmethod
    def init_model(self, checkpoint=None):
        pass

    @abstractmethod
    def predict(self, image):
        pass

