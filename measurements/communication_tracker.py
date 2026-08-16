import time

class CommunicationTracker:
    def __init__(self):
        self.bytes_sent = 0
        self.bytes_received = 0
        self.message_sent = 0
        self.message_received = 0
        self.comm_time = 0.0
        self._comm_start_time = None

    def record_sent(self, message: bytes):
        self.bytes_sent += len(message)
        self.message_sent += 1

    def record_received(self, message: bytes):
        self.bytes_received += len(message)
        self.message_received += 1

    def timer_start(self):
        self._comm_start_time = time.time()

    def timer_stop(self):
        if self._comm_start_time is not None:
            elapsed_time = time.time() - self._comm_start_time
            self.comm_time += elapsed_time
            self._comm_start_time = None

    def get_recordings(self):
        row = {
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "message_sent": self.message_sent,
            "message_received": self.message_received,
            "comm_time (sec)": self.comm_time
        }

        self.bytes_sent = 0
        self.bytes_received = 0
        self.message_sent = 0
        self.message_received = 0
        self.comm_time = 0.0

        return row