from abc import ABC, abstractmethod

class PrivacyProtocol(ABC):
    @abstractmethod
    def before_send(self, weights: dict) -> dict:
        # apply privacy protocol before submitting the weights to semi-honest peers
        pass

    @abstractmethod
    def after_receive(self, weights: dict) -> dict:
        # prepare received weights (encrypted with privacy protocols) before aggregation
        pass