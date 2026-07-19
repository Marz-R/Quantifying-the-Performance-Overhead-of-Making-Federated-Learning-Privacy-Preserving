import psutil
import GPUtil
import threading
import numpy as np
import os

class HardwareTracker:
    def __init__(self, sampling_interval: float = 0.5):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self.sampling_interval = sampling_interval

        self.gpu_list = [gpu for gpu in GPUtil.getGPUs()]
        self.cpu_count = psutil.cpu_count(logical=True) or 1
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = {}

        self._process = psutil.Process(os.getpid()) # only measure the usage of main process, i.e. PeerNode.training
        self._process.cpu_percent(interval=None) # internal baseline, discard


    def _recording(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(self.sampling_interval)
            if self._stop_event.is_set():
                break

            with self._lock:
                self.cpu_usage.append(self._process.cpu_percent(interval=None) / self.cpu_count)
                self.memory_usage.append(self._process.memory_info().rss / 1e6)

                for gpu in self.gpu_list:
                    if gpu.id not in self.gpu_usage:
                        self.gpu_usage[gpu.id] = []
                    self.gpu_usage[gpu.id].append(gpu.load * 100)


    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._recording)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

        # wait for child thread to finish before returning
        if self._thread is not None:
            self._thread.join(timeout=self.sampling_interval * 2)


    def get_usage(self):
        with self._lock:
            row = {
                "cpu_mean (%)": np.mean(self.cpu_usage),
                "cpu_max (%)": np.max(self.cpu_usage),
                "memory_mean (mb)": np.mean(self.memory_usage),
                "memory_max (mb)": np.max(self.memory_usage)
            }

            for gpu_id, usage in self.gpu_usage.items():
                row[f"gpu{gpu_id}_mean (%)"] = np.mean(usage)
                row[f"gpu{gpu_id}_max (%)"] = np.max(usage)

            self.cpu_usage = []
            self.memory_usage = []
            self.gpu_usage = {}

        return row