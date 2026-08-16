import base64
import time
import io
import queue
from typing import Optional
from enum import Enum, auto
from tqdm import tqdm
from p2pnetwork.node import Node
import torch
import torch.nn as nn
from torchmetrics.classification import F1Score
from bulletin_client import BulletinClient
from data_load import load_data
from early_stopping import EarlyStopping
from measurements.exp_logger import ExperimentLogger
from measurements.communication_tracker import CommunicationTracker
from measurements.hardware_tracker import HardwareTracker


# Hyperparameters
VALID_SPLIT = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RoundState(Enum):
    IDLE = auto()
    TRAINING = auto()
    COMMUNICATION = auto()
    AGGREGATION = auto()
    CONSENSUS = auto()


class PeerNode (Node):
    def __init__(self, 
                 host: str, 
                 port: int, 
                 model: nn.Module, 
                 dataset: str,
                 exp_logger: ExperimentLogger,
                 privacy_protocol: str,
                 id=None, # id is *string*, if input is int, parent class will convert it to string
                 batch_size=128,
                 sync_every=5,
                 callback=None, 
                 max_connections=0, # 0 means unlimited connections
                 debug_message=False):
        super(PeerNode, self).__init__(host, port, id, callback, max_connections)
        
        self.state = RoundState.IDLE

        self.batch_size = batch_size
        self.model = model.to(device)
        self.train_data, self.val_data = load_data(dataset, self.id, self.batch_size, VALID_SPLIT)

        self.iteration = 0
        self.sync_every = sync_every # default 5

        self.privacy_protocol = privacy_protocol

        self.bulletin: Optional[BulletinClient] = None

        self.peer_list = {} # peer_id: {"host": str, "port": int}
        self.last_seen_version = 0
        self.connected_peers = []

        self.max_epochs = None
        self.early_stopping = EarlyStopping(patience=10, min_delta=0.0001)

        self.peers_weights = {} # iteration: {peer_id: weights}

        self.weights_queue = queue.Queue()
        self.ready_queue = queue.Queue()

        self.logger = exp_logger
        self.comm_tracker = CommunicationTracker()
        self.hardware_tracker = HardwareTracker(sampling_interval=0.5)

        self.debug_message = debug_message


    def print_debug_messages(self, message):
        if self.debug_message:
            print("**DEBUG** Node " + self.id + " : " + message)


    def tensor_to_base64(self, tensor: torch.Tensor) -> dict:
        buffer = io.BytesIO()
        torch.save(tensor.cpu(), buffer)
        return {
            "_type":   "tensor",
            "data":    base64.b64encode(buffer.getvalue()).decode("utf-8"),
            "dtype":   str(tensor.dtype),
            "shape":   list(tensor.shape),
        }


    def base64_to_tensor(self, encoded: dict) -> torch.Tensor:
        raw = base64.b64decode(encoded["data"].encode("utf-8"))
        buffer = io.BytesIO(raw)
        return torch.load(buffer, map_location="cpu", weights_only=False)


    def node_message(self, node, data):
        self.comm_tracker.record_received(data)

        if node.id == str(data["sender_id"]): # is it nessary to check sender_id?

            if data["type"] == "weights_submission":
                self.handle_weights_submission(node, data)

            elif data["type"] == "iteration_ready":
                self.print_debug_messages("Received iteration ready from node " + str(data["sender_id"]) + " for iteration " + str(data["iteration"]))
                self.ready_queue.put(data)

            elif data["type"] == "test_message":
                self.print_debug_messages("Received test message: " + data["message"] + " from node " + str(data["sender_id"]) + " to node " + str(self.id))
            
            else:
                self.print_debug_messages("Received message with unknown type: " + data["type"] + " from node " + node.id)
       
        else:
            self.print_debug_messages("Received message with mismatching sender_id: " + str(data["sender_id"]) + " from node " + node.id)


    def register_to_network(self, bulletin):
        # Register self to the public bulletin
        self.bulletin = bulletin
        self.bulletin.add_peer(self.id, {"host": self.host, "port": self.port})

        peer_list_response = self.bulletin.get_peer_list()
        self.peer_list = peer_list_response["peer_list"]
        self.last_seen_version = peer_list_response["version"]

        self.max_epochs = self.bulletin.get_num_epochs()

        self.print_debug_messages("Registered to network.")

    
    def connect_with_peers(self):
        # Connect to all peers in the peer list
        ids = [n.id for n in self.nodes_outbound] + [n.id for n in self.nodes_outbound]

        for peer_id, peer_info in self.peer_list.items():
            if peer_id != self.id and peer_id not in ids:
                self.connect_with_node(peer_info["host"], peer_info["port"])
                self.print_debug_messages("Connected to Peer " + peer_id)

        self.connected_peers = self.nodes_outbound + self.nodes_inbound


    def update_peer_list(self):
        response = self.bulletin.update_peer_list(self.last_seen_version)

        if response["peer_list"] is not None:
            self.peer_list = response["peer_list"]
            self.last_seen_version = response["version"]

            for peer in self.connected_peers:
                if peer.id not in self.peer_list:
                    self.disconnect_with_node(peer)
                    self.print_debug_messages("Peer " + peer.id + " left, disconnected.")
                    self.connected_peers.remove(peer)

            self.connect_with_peers()
        

    def _disconnect_all_peers(self):
        for node in self.connected_peers:
            self.disconnect_with_node(node)
            self.connected_peers.remove(node)


    def quit_network(self):
        self.bulletin.remove_peer(self.id)

        self.bulletin = None
        self._disconnect_all_peers()
        self.peer_list = {}
        self.peers_weights = {}


    def training(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, weight_decay = 0.005, momentum = 0.9)
        
        self.print_debug_messages("Starting training")
        
        f1 = F1Score(task='multiclass', num_classes=10, average='macro').to(device)

        epoch_bar = tqdm(range(self.max_epochs), desc="Training Progress", leave=False)

        for epoch in epoch_bar:
            self.update_peer_list()

            self.print_debug_messages("Starting epoch " + str(epoch+1))

            self.model.train()
            #torch.autograd.set_detect_anomaly(True)
            total_train_loss = 0.0
            total_training_samples = 0            

            train_bar = tqdm(self.train_data, desc=f"Epoch {epoch+1}/{self.max_epochs} [training]", leave=False)
            epoch_start_time = time.time()
            num_batches = len(self.train_data)

            self.hardware_tracker.start()

            # training loop
            for i, (images, labels) in enumerate(train_bar):  
                self.state = RoundState.TRAINING
                self.iteration = i
                self.print_debug_messages("Training iteration " + str(i))

                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()

                self.hardware_tracker.phase_start("gradient_computation")
                train_outputs = self.model(images)
                train_loss = criterion(train_outputs, labels)
                
                train_loss.backward()
                optimizer.step()
                self.hardware_tracker.phase_stop("gradient_computation")

                sample_size = images.size(0)
                total_train_loss += train_loss.item() * sample_size
                total_training_samples += sample_size

                # (always sync on the final batch of the epoch so peers end each epoch aligned)
                is_sync_iteration = ((i + 1) % self.sync_every == 0) or (i == num_batches - 1)

                if not is_sync_iteration:
                    self.print_debug_messages("Skipping sync for iteration " + str(i))
                    continue

                # get model weights and save to history
                state_dict = self.model.state_dict()
                cpu_state_dict = {k: v.cpu().clone().detach() for k, v in state_dict.items()}
                #private_state_dict = self.apply_privacy_protocol(cpu_state_dict)

                # encode weights and submit to peers
                self.comm_tracker.timer_start()
                self.submit_weights(cpu_state_dict)

                # collect weights from peers for current iteration before aggregation
                expected_weights_count = len(self.peer_list) - 1 # excluding self
                self.peers_weights[i] = {}

                while len(self.peers_weights[i]) < expected_weights_count:
                    try:
                        sender_id, message = self.weights_queue.get(timeout=60)
                        if message["iteration"] == i:
                            self.peers_weights[i][sender_id] = message["weights"]
                        elif message["iteration"] > i: # avoid re-queueing
                            self.peers_weights.setdefault(message["iteration"], {})[sender_id] = message["weights"]
                    except queue.Empty:
                        self.print_debug_messages("Timeout waiting for weights, iteration " + str(i))
                        break

                # update local model with aggregated weights from peers
                if i in self.peers_weights:
                    self.print_debug_messages("Updating local model with weights from iteration " + str(i))
                    aggregated_weights = self.aggregate_weights(self.model.state_dict(), self.peers_weights[i])
                    self.model.load_state_dict(aggregated_weights)
                    del self.peers_weights[i] # clear the weights for this iteration after aggregation

                self.print_debug_messages("Finished training iteration " + str(i))

                self.iteration_ready(i)
                self.comm_tracker.timer_stop()


            total_val_losses = 0.0
            total_val_samples = 0

            val_bar = tqdm(self.val_data, desc=f"Epoch {epoch+1}/{self.max_epochs} [validation]", leave=False)

            # validation loop
            with torch.no_grad():

                self.model.eval()

                for i, (images, labels) in enumerate(val_bar):
                    images = images.to(device)
                    labels = labels.to(device)

                    val_outputs = self.model(images)
                    val_loss = criterion(val_outputs, labels)

                    sample_size = images.size(0)
                    total_val_losses += val_loss.item() * sample_size
                    total_val_samples += sample_size

                    f1.update(val_outputs, labels)

            self.hardware_tracker.stop()

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            itr_per_sec = num_batches / epoch_duration if epoch_duration > 0 else 0

            avg_train_loss = total_train_loss / total_training_samples if total_training_samples > 0 else 0
            avg_val_loss = total_val_losses / total_val_samples if total_val_samples > 0 else 0
            val_f1 = f1.compute()
            f1.reset()

            tqdm.write(
                f"Peer {self.id} --- "
                f"Epoch {epoch+1}/{self.max_epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={avg_val_loss:.4f} | "
                f"val_f1={val_f1:.4f}"
            )

            self.logger.log_communication(epoch+1, self.max_epochs, len(self.peer_list), self.comm_tracker.get_recordings())
            self.logger.log_computation(epoch+1, self.max_epochs, len(self.peer_list), self.hardware_tracker.get_usage())

            # check for early stopping
            if self.early_stopping.stop(avg_val_loss):
                self.logger.log_performance(epoch+1, self.batch_size, len(self.peer_list), avg_train_loss, avg_val_loss, val_f1.item(), itr_per_sec, True)
                self.print_debug_messages("Early stopping triggered at epoch " + str(epoch))
                break
            
            self.logger.log_performance(epoch+1, self.max_epochs, len(self.peer_list), avg_train_loss, avg_val_loss, val_f1.item(), itr_per_sec, False)

        self.print_debug_messages("Finished training")


    def apply_privacy_protocol(self, state_dict):
        if self.privacy_protocol == "Plaintext":
            return state_dict


    def submit_weights(self, weights, recipient_id=None, recipient_host=None):
        self.state = RoundState.COMMUNICATION

        # encode weights to base64 for transmission
        encoded_state_dict = {k: self.tensor_to_base64(v) for k, v in weights.items()}
        message = {
            "type": "weights_submission",
            "sender_id": self.id,
            "timestamp": time.time(),
            "iteration": self.iteration,
            "batch_size": self.batch_size,
            "weights":  encoded_state_dict
        }

        # send corresponding weights to peers using outbound connections for security
        for node in self.connected_peers:
            if recipient_id is None and recipient_host is None:
                self.send_to_node(node, message)
                self.comm_tracker.record_sent(message)
                self.print_debug_messages("Submitted weights to node " + node.id)
            else:
                if node.id == recipient_id and node.host == recipient_host:
                    self.send_to_node(node, message)
                    self.comm_tracker.record_sent(message)
                    self.print_debug_messages("Submitted weights to node " + node.id)
                    break


    def handle_weights_submission(self, node, data):
        self.print_debug_messages("Received weights from " + node.id)

        message = data
        message["weights"]= {k: self.base64_to_tensor(v) for k, v in message["weights"].items()}

        self.weights_queue.put((node.id, message))


    def aggregate_weights(self, local_state_dict, peers_weights):
        self.print_debug_messages("Aggregating...")

        all_weights = [local_state_dict] + list(peers_weights.values())
        aggregated_weights = {}

        for key in local_state_dict.keys():
            aggregated_weights[key] = torch.stack([w[key].float().to(device) for w in all_weights]).mean(dim=0).detach().clone()
        
        return aggregated_weights
        

    def iteration_ready(self, iteration):
        self.state = RoundState.COMMUNICATION

        message = {
            "type": "iteration_ready",
            "sender_id": self.id,
            "timestamp": time.time(),
            "iteration": iteration
        }

        for node in self.connected_peers:
            self.send_to_node(node, message)
            self.comm_tracker.record_sent(message)
        
        expected_ready_count = len(self.peer_list) - 1
        ready_peers = []

        while len(ready_peers) < expected_ready_count:
            try:
                message = self.ready_queue.get(timeout=60)
                if message["iteration"] >= iteration and message["sender_id"] not in ready_peers:
                    ready_peers.append(message["sender_id"])
            except queue.Empty:
                self.print_debug_messages("Timeout waiting for ready signals, iteration " + str(iteration))
                break