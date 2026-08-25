from privacy_protocol import PrivacyProtocol

class SecureAggregation(PrivacyProtocol):
    def __init__(self):
        pass

    def before_send(self, weights: dict) -> dict:
        pass

    def after_receive(self, weights: dict) -> dict:
        pass