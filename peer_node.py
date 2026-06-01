import base64
import time
import io
import json
import threading
from typing import Optional
from enum import Enum, auto
from p2pnetwork.node import Node
import torch
import torch.nn as nn
from bulletin import Public_Bulletin, Observer
from data_load import load_dataset


# Hyperparameters
NUM_EPOCHS = 20
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
                 max_connections=0,):
        super(PeerNode, self).__init__(host, port, id, callback, max_connections)
        
        self.state = RoundState.IDLE

        self.model = model.to(device)
        self.dataset = load_dataset(dataset, BATCH_SIZE, VALID_SPLIT, download=False)[0] # only load train dataset

        self.iteration = 0
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=INIT_LR, weight_decay = 0.005, momentum = 0.9)

        self.bulletin: Optional[Public_Bulletin] = None
        self.peer_list = {} # peer_id: {"host": str, "port": int}
        self.connected_peers = {}

        self.history = []
        self.peers_weights = {} # iteration: {peer_id: weights}

        self.lock = threading.Lock()
        self.barrier = threading.Barrier(len(self.peer_list) + 1) # +1 for self; TODO : update barrier when new peer joins or leaves


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
        print("Node " + self.id + ": Registered to network.")

        self.bulletin.subscribe(self)

        self.peer_list = self.bulletin.get_peer_list()

    
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
        print("Node " + self.id + ": Starting training")
        
        self.model.train()

        # Load in the data in batches
        for i, (images, labels) in enumerate(self.dataset):  
            self.state = RoundState.TRAINING
            self.iteration = i
            print("Node " + self.id + ": Training iteration " + str(i))

            # update local model with aggregated weights from peers last iteration (if exist)
            if i-1 in self.peers_weights:
                print("Node " + self.id + ": Updating local model with weights from iteration " + str(i-1))
                aggregated_weights = self.aggregate_weights(self.model.state_dict(), self.peers_weights[i-1])
                self.model.load_state_dict(aggregated_weights)

            images = images.to(device)
            labels = labels.to(device)
                
            train_outputs = self.model(images)
            train_loss = self.criterion(train_outputs, labels)
                
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            # get model weights and save to history
            state_dict = self.model.state_dict()
            cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
            
            snapshot = {
                "iteration": i,
                "timestamp": time.time(),
                "weights":  cpu_state_dict
            }
            self.history.append(snapshot)

            # encode weights and submit to peers
            self.submit_weights(cpu_state_dict)

            # wait for all peers to submit weights before next iteration
            self.barrier.wait() 

            if i>=3:
                break # only 3 batches for testing


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

        self.peers_weights[message["iteration"]] = {node.id: message["weights"]}


    def aggregate_weights(self, local_state_dict, peers_weights):
        print("Node " + self.id + ": Aggregating...")

        all_weights = [local_state_dict] + list(peers_weights.values())
        aggregated_weights = {}

        for key in local_state_dict.keys():
            aggregated_weights[key] = torch.stack([w[key].float() for w in all_weights]).mean(dim=0)
        
        return aggregated_weights
        

    def voting_consensus(self):
        print("Node " + self.id + ": Performing voting consensus")

    def update_model(self):
        print("Node " + self.id + ": Updating model")

    def get_connected_peers(self):
        return self.connected_peers