import os
import csv
import pandas as pd
import numpy as np
import psutil
import GPUtil

class ExperimentLogger:
    def __init__(self, peer_id, model, dataset, privacy_protocol, output_path='./measurements'):
        self.peer_id = peer_id
        
        self.csv_path = output_path
        self.csv_name_perf = peer_id + "_performance_data_" + privacy_protocol + "_" + model + "model_" + dataset + ".csv"
        self.csv_name_comp = peer_id + "_computation_data_" + privacy_protocol + "_" + model + "model_" + dataset + ".csv"
        self.csv_name_comm = peer_id + "_communication_data_" + privacy_protocol + "_" + model + "model_" + dataset + ".csv"
        
        self.bytes_sent = 0
        self.bytes_received = 0
        self.message_sent = 0
        self.message_received = 0

        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = {}


    def log_performance(self, epoch, max_epoch, num_peers, batch_size, train_loss, val_loss, val_f1, itr_per_sec, convergence: bool):
        row = {
            "peer_id": self.peer_id,
            "epoch": epoch, 
            "max_epoch": max_epoch,
            "num_peers": num_peers,
            "batch_size": batch_size,
            "train_losses": train_loss,
            "validation_losses": val_loss,
            "validation_f1_scores": val_f1,
            "iterations_per_second": itr_per_sec,
            "convergence": convergence
        }

        path = os.path.join(self.csv_path, self.csv_name_perf)
        df = pd.DataFrame(row)
        df.to_csv(path, index=False, mode='a', header=not os.path.exists(path))


    def log_computation(self, epoch, max_epoch, num_peers, batch_size): # log after an epoch
        row = {
            "peer_id": self.peer_id,
            "epoch": epoch,
            "max_epoch": max_epoch,
            "num_peers": num_peers,
            "batch_size": batch_size,
            "cpu_mean": np.mean(self.cpu_usage),
            "cpu_max": np.max(self.cpu_usage),
            "memory_mean": np.mean(self.memory_usage),
            "memory_max": np.max(self.memory_usage)
        }

        for gpu_id, usage in self.gpu_usage.items():
            row[f"gpu{gpu_id}_mean"] = np.mean(usage)
            row[f"gpu{gpu_id}_max"] = np.max(usage)

        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = {}

        path = os.path.join(self.csv_path, self.csv_name_comp)
        df = pd.DataFrame(row)
        df.tocsv(path, index=False, mode='a', header=not os.path.exists(path))
        

    def log_communication(self, epoch, max_epoch, num_peers, batch_size): # log communication after finishing an epoch
        row = {
            "peer_id": self.peer_id,
            "epoch": epoch,
            "max_epoch": max_epoch,
            "num_peers": num_peers,
            "batch_size": batch_size,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "message_sent": self.message_sent,
            "message_received": self.message_received,
        }

        self.bytes_sent = 0
        self.bytes_received = 0
        self.message_sent = 0
        self.message_received = 0

        path = os.path.join(self.csv_path, self.csv_name_comm)
        df = pd.DataFrame(row)
        df.tocsv(path, index=False, mode='a', header=not os.path.exists(path))


    def record_hardware_usage(self): # record for every iteration or every fixed time interval
        self.cpu_usage.append(psutil.cpu_percent())
        self.memory_usage.append(psutil.virtual_memory().percent)

        gpus = GPUtil.getGPUs()
        for gpu in gpus: # TODO figure out how many gpus are in a server, maybe we can just skip this if only 1 gpu is there
            if gpu.id not in self.gpu_usage:
                self.gpu_usage[gpu.id] = []
            self.gpu_usage[gpu.id].append(gpu.load * 100)


    def record_sent(self, message: bytes):
        self.bytes_sent += len(message)
        self.message_sent += 1


    def record_received(self, message: bytes):
        self.bytes_received += len(message)
        self.message_received += 1