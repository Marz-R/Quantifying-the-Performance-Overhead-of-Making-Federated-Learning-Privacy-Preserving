import os
import pandas as pd
import numpy as np

class ExperimentLogger:
    def __init__(self, peer_id, model, dataset, privacy_protocol, output_path='./measurements'):
        self.peer_id = peer_id
        
        self.csv_path = output_path
        self.csv_name_perf = peer_id + "_performance_data_" + privacy_protocol + "_" + model + "model_" + dataset + ".csv"
        self.csv_name_comp = peer_id + "_computation_data_" + privacy_protocol + "_" + model + "model_" + dataset + ".csv"
        self.csv_name_comm = peer_id + "_communication_data_" + privacy_protocol + "_" + model + "model_" + dataset + ".csv"


    def log_performance(self, epoch, max_epoch, num_peers, train_loss, val_loss, val_f1, itr_per_sec, convergence: bool):
        row = {
            "peer_id": self.peer_id,
            "epoch": epoch, 
            "max_epoch": max_epoch,
            "num_peers": num_peers,
            "train_losses": train_loss,
            "validation_losses": val_loss,
            "validation_f1_scores": val_f1,
            "iterations_per_second": itr_per_sec,
            "convergence": convergence
        }

        path = os.path.join(self.csv_path, self.csv_name_perf)
        df = pd.DataFrame([row])
        df.to_csv(path, index=False, mode='a', header=not os.path.exists(path))


    def log_computation(self, epoch, max_epoch, num_peers, comp_row: dict): # log after an epoch
        row = {
            "peer_id": self.peer_id,
            "epoch": epoch,
            "max_epoch": max_epoch,
            "num_peers": num_peers,
            **comp_row
        }

        path = os.path.join(self.csv_path, self.csv_name_comp)
        df = pd.DataFrame([row]) # single row, so wrap in a list to create a DataFrame
        df.to_csv(path, index=False, mode='a', header=not os.path.exists(path))
        

    def log_communication(self, epoch, max_epoch, num_peers, comm_row: dict): # log communication after finishing an epoch
        row = {
            "peer_id": self.peer_id,
            "epoch": epoch,
            "max_epoch": max_epoch,
            "num_peers": num_peers,
            **comm_row
        }

        path = os.path.join(self.csv_path, self.csv_name_comm)
        df = pd.DataFrame([row])
        df.to_csv(path, index=False, mode='a', header=not os.path.exists(path))
