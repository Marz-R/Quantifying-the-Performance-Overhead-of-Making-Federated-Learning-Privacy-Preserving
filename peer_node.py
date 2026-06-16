import base64
import time
import io
import json
import queue
from typing import Optional
from enum import Enum, auto
from p2pnetwork.node import Node
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from bulletin import Public_Bulletin, Observer
from data_load import load_dataset
from early_stopping import EarlyStopping


# Hyperparameters
INIT_LR = 0.001
BATCH_SIZE = 64
VALID_SPLIT = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RoundState(Enum):
    IDLE = auto()
    TRAINING = auto()
    COMMUNICATION = auto()
    AGGREGATION = auto()
    CONSENSUS = auto()


class PeerNode (Node, Observer):
    def __init__(self, 
                 host: str, 
                 port: int, 
                 model: nn.Module, 
                 dataset: str,
                 id=None, # id is *string*, if input is int, parent class will convert it to string
                 callback=None, 
                 max_connections=0,
                 debug=False):
        super(PeerNode, self).__init__(host, port, id, callback, max_connections)
        
        self.state = RoundState.IDLE

        self.model = model.to(device)
        self.train_data, self.val_data, _ = load_dataset(dataset, BATCH_SIZE, VALID_SPLIT, download=False)

        self.iteration = 0

        self.bulletin: Optional[Public_Bulletin] = None
        self.peer_list = {} # peer_id: {"host": str, "port": int}
        self.connected_peers = {}
        self.max_epochs = 50
        self.early_stopping = EarlyStopping(patience=10, min_delta=0.0001)

        self.history = []
        self.peers_weights = {} # iteration: {peer_id: weights}

        self.weights_queue = queue.Queue()
        self.ready_queue = queue.Queue()

        self.debug = debug


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
        if node.id == str(data["sender_id"]): # is it nessary to check sender_id?

            if data["type"] == "weights_submission":
                self.handle_weights_submission(node, data)

            elif data["type"] == "iteration_ready":
                print("Node " + self.id + ": Received iteration ready from node " + str(data["sender_id"]) + " for iteration " + str(data["iteration"]))
                self.ready_queue.put(data)

            elif data["type"] == "test_message":
                print("Node " + self.id + ": Received test message: " + data["message"] + " from node " + str(data["sender_id"]) + " to node " + str(self.id))
            
            else:
                print("Node " + self.id + ": Received message with unknown type: " + data["type"] + " from node " + node.id)
       
        else:
            print("Node " + self.id + ": Received message with mismatching sender_id: " + str(data["sender_id"]) + " from node " + node.id)


    def register_to_network(self, bulletin):
        # Register self to the public bulletin
        self.bulletin = bulletin
        self.bulletin.add_peer(self.id, {"host": self.host, "port": self.port})
        self.bulletin.subscribe(self)
        self.peer_list = self.bulletin.get_peer_list()
        self.max_epochs = self.bulletin.get_num_epochs()

        print("Node " + self.id + ": Registered to network.")

    
    def connect_with_peers(self):
        # Connect to all peers in the peer list
        for peer_id, peer_info in self.peer_list.items():
            if peer_id != self.id:
                self.connect_with_node(peer_info["host"], peer_info["port"])
                self.connected_peers[peer_id] = peer_info
                print("Node " + self.id + ": Connected to Peer " + peer_id)

    def _disconnect_with_peers(self):
        for peer_id, peer_info in self.peer_list.items():
            if peer_id != self.id:
                self.disconnect_with_node(peer_info["host"], peer_info["port"])
                del self.connected_peers[peer_id]
                print("Node " + self.id + ": Disconnected to Peer " + peer_id)


    def on_add_peer(self, peer_id: int, peer_info: dict):
        if peer_id != self.id: 
            self.peer_list[peer_id] = peer_info
            print("Node " + self.id + ": Updated peer list: Peer " + peer_id + " joined.")

    def on_remove_peer(self, peer_id: int):
        if peer_id in self.peer_list:
            del self.peer_list[peer_id]
            print("Node " + self.id + ": Updated peer list: Peer " + peer_id + " left.")


    def quit_network(self):
        self.bulletin.remove_peer(self.id)
        self.bulletin.unsubscribe(self)

        self.bulletin = None
        self._disconnect_with_peers()
        self.peer_list = {}
        self.peers_weights = {}


    def training(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=INIT_LR, weight_decay = 0.005, momentum = 0.9)
        
        print("\nNode " + self.id + ": Starting training")

        for epoch in range(self.max_epochs):
            print("\nNode " + self.id + ": Starting epoch " + str(epoch))

            self.model.train()
            #torch.autograd.set_detect_anomaly(True)
            total_train_loss = 0
            total_training_samples = 0
            train_loss_list = []

            # training loop
            for i, (images, labels) in enumerate(self.train_data):  
                self.state = RoundState.TRAINING
                self.iteration = i
                print("Node " + self.id + ": Training iteration " + str(i))

                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                train_outputs = self.model(images)
                train_loss = criterion(train_outputs, labels)
                
                train_loss.backward()
                optimizer.step()

                sample_size = images.size(0)
                total_train_loss += train_loss.item() * sample_size
                total_training_samples += sample_size

                # get model weights and save to history
                state_dict = self.model.state_dict()
                cpu_state_dict = {k: v.cpu().clone().detach() for k, v in state_dict.items()}
                
                snapshot = {
                    "iteration": i,
                    "timestamp": time.time(),
                    "weights":  cpu_state_dict
                }
                self.history.append(snapshot)

                # encode weights and submit to peers
                self.submit_weights(cpu_state_dict)

                # collect weights from peers for current iteration before aggregation
                expected_weights_count = len(self.peer_list) - 1 # excluding self
                self.peers_weights[i] = {}

                while len(self.peers_weights[i]) < expected_weights_count:
                    try:
                        sender_id, message = self.weights_queue.get(timeout=60)
                        if message["iteration"] == i:
                            self.peers_weights[i][sender_id] = message["weights"]
                        elif message["iteration"] > i:
                            self.weights_queue.put((sender_id, message)) # Re-queue the message from furture iteration
                    except queue.Empty:
                        print("Node " + self.id + ": Timeout while waiting for weights from peers for iteration " + str(i))
                        break

                # update local model with aggregated weights from peers
                if i in self.peers_weights:
                    print("Node " + self.id + ": Updating local model with weights from iteration " + str(i))
                    aggregated_weights = self.aggregate_weights(self.model.state_dict(), self.peers_weights[i])
                    self.model.load_state_dict(aggregated_weights)

                print("Node " + self.id + ": Finished training iteration " + str(i))

                self.iteration_ready(i)


            total_val_losses = 0
            total_val_samples = 0
            val_loss_list = []

            # validation loop
            with torch.no_grad():

                self.model.eval()

                for i, (images, labels) in enumerate(self.val_data):
                    images = images.to(device)
                    labels = labels.to(device)

                    val_outputs = self.model(images)
                    val_loss = criterion(val_outputs, labels)

                    sample_size = images.size(0)
                    total_val_losses += val_loss.item() * sample_size
                    total_val_samples += sample_size

            #avg_train_loss = total_train_loss / total_training_samples if total_training_samples > 0 else 0
            avg_val_loss = total_val_losses / total_val_samples if total_val_samples > 0 else 0

            #train_loss_list.append(avg_train_loss)
            val_loss_list.append(avg_val_loss)

            # check for early stopping
            if self.early_stopping.stop(avg_val_loss):
                print("Node " + self.id + ": Early stopping triggered at epoch " + str(epoch))
                break
        
        print("Node " + self.id + ": Finished training")
        self.plot_convergence(train_loss_list, val_loss_list)


    def plot_convergence(self, train_loss, val_loss):
        epochs = range(1, len(train_loss) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    def submit_weights(self, weights, recipient_id=None, recipient_host=None):
        self.state = RoundState.COMMUNICATION

        # encode weights to base64 for transmission
        encoded_state_dict = {k: self.tensor_to_base64(v) for k, v in weights.items()}
        message = {
            "type": "weights_submission",
            "sender_id": self.id,
            "timestamp": time.time(),
            "iteration": self.iteration,
        "batch_size": BATCH_SIZE,
            "weights":  encoded_state_dict
        }

        # send corresponding weights to peers using outbound connections for security
        for node in self.nodes_outbound:
            if recipient_id is None and recipient_host is None:
                self.send_to_node(node, message)
                print("Node " + self.id + ": Submitted weights to node " + node.id)
            else:
                if node.id == recipient_id and node.host == recipient_host:
                    self.send_to_node(node, message)
                    print("Node " + self.id + ": Submitted weights to node " + node.id)
                    break


    def handle_weights_submission(self, node, data):
        print("Node " + self.id + ": Received weights from " + node.id)

        message = json.loads(json.dumps(data))
        message["weights"]= {k: self.base64_to_tensor(v) for k, v in message["weights"].items()}

        self.weights_queue.put((node.id, message))


    def aggregate_weights(self, local_state_dict, peers_weights):
        print("Node " + self.id + ": Aggregating...")

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

        for node in self.nodes_outbound:
            self.send_to_node(node, message)
        
        expected_ready_count = len(self.peer_list) - 1
        ready_peers = []

        while len(ready_peers) < expected_ready_count:
            try:
                message = self.ready_queue.get(timeout=60)
                if message["iteration"] == iteration:
                    ready_peers.append(message["sender_id"])
                else:
                    self.ready_queue.put(message) # put it back if it's for a different iteration
            except queue.Empty:
                print("Node " + self.id + ": Timeout while waiting for ready signals for iteration " + str(iteration))
                break


    def voting_consensus(self):
        print("Node " + self.id + ": Performing voting consensus")


    def get_connected_peers(self):
        return self.connected_peers