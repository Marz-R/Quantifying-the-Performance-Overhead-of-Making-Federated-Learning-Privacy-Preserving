class CommunicationTracker:
    def __init__(self):
        self.bytes_sent = 0
        self.bytes_received = 0
        self.message_sent = 0
        self.message_received = 0

    def record_sent(self, message: bytes):
        self.bytes_sent += len(message)
        self.message_sent += 1

    def record_received(self, message: bytes):
        self.bytes_received += len(message)
        self.message_received += 1

    def get_recordings(self):
        row = {
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "message_sent": self.message_sent,
            "message_received": self.message_received,
        }

        self.bytes_sent = 0
        self.bytes_received = 0
        self.message_sent = 0
        self.message_received = 0

        return row