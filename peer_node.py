import time
import queue
from typing import Optional
from enum import Enum, auto
from tqdm import tqdm
from p2p_node import Node
import torch
import torch.nn as nn
from torchmetrics.classification import F1Score
from bulletin_client import BulletinClient
from data_load import load_data
from early_stopping import EarlyStopping
from measurements.exp_logger import ExperimentLogger
from measurements.communication_tracker import CommunicationTracker
from measurements.hardware_tracker import HardwareTracker
from privacy_adss import AdditiveSecretSharing
from flat import Flat


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
                 partition_type="iid",
                 debug_message=False):
        super(PeerNode, self).__init__(host, port, id)
        
        self.state = RoundState.IDLE

        self.batch_size = batch_size
        self.model = model.to(device)
        self.train_data, self.val_data = load_data(dataset, self.id, self.batch_size, VALID_SPLIT, partition_type)

        self.iteration = 0
        self.sync_every = sync_every # default 5

        self.bulletin: Optional[BulletinClient] = None

        self.peer_list = {} # peer_id: {"host": str, "port": int}
        self.last_seen_version = 0

        self.max_epochs = None
        self.early_stopping = EarlyStopping(patience=10, min_delta=0.0001)

        self.peers_weights = {} # iteration: {peer_id: weights}, iteration is (epoch, batch index) or (epoch, "end")

        self.weights_queue = queue.Queue()

        self.logger = exp_logger
        self.comm_tracker = CommunicationTracker()
        self.hardware_tracker = HardwareTracker(sampling_interval=0.5)

        if privacy_protocol == "AdditiveSecretSharing":
            self.privacy_protocol = AdditiveSecretSharing(peers_list=[]) 
            self.partial_weights = {}
            self.partial_weights_queue = queue.Queue()
        else:
            self.privacy_protocol = None

        self.flat_layout = None # cached state_dict layout for the plaintext path

        self.debug_message = debug_message


    def print_debug_messages(self, message):
        if self.debug_message:
            print("**DEBUG** Node " + self.id + " : " + message)


    def node_message(self, node_id, data, num_bytes=0):
        self.comm_tracker.record_received(num_bytes)

        if node_id == str(data["sender_id"]): # is it nessary to check sender_id?

            if data["type"] == "weights_submission":
                self.handle_weights_submission(node_id, data)

            elif data["type"] == "test_message":
                self.print_debug_messages("Received test message: " + data["message"] + " from node " + str(data["sender_id"]) + " to node " + str(self.id))
            
            else:
                self.print_debug_messages("Received message with unknown type: " + data["type"] + " from node " + node_id)
       
        else:
            self.print_debug_messages("Received message with mismatching sender_id: " + str(data["sender_id"]) + " from node " + node_id)


    def register_to_network(self, bulletin):
        # Register self to the public bulletin
        self.bulletin = bulletin
        self.bulletin.add_peer(self.id, {"host": self.host, "port": self.port})

        peer_list_response = self.bulletin.get_peer_list()
        self.peer_list = peer_list_response["peer_list"]
        self.last_seen_version = peer_list_response["version"]

        if isinstance(self.privacy_protocol, AdditiveSecretSharing):
            self.privacy_protocol.update_peers_list([peer_id for peer_id in self.peer_list.keys() if peer_id != self.id])

        self.max_epochs = self.bulletin.get_num_epochs()

        self.print_debug_messages("Registered to network.")

    
    def connect_with_peers(self):
        # Connect to all peers in the peer list
        with self._lock:
            peer_ids = list(self.connected_nodes.keys())
        for peer_id, peer_info in self.peer_list.items():
            if peer_id != self.id and peer_id not in peer_ids:
                self.connect_with_node(peer_info["host"], peer_info["port"])
                self.print_debug_messages("Connected to Peer " + peer_id)


    def update_peer_list(self):
        response = self.bulletin.update_peer_list(self.last_seen_version)

        if response["peer_list"] is not None:
            self.peer_list = response["peer_list"]
            self.last_seen_version = response["version"]

            with self._lock:
                peer_ids = list(self.connected_nodes.keys())
            for peer_id in peer_ids:
                if peer_id not in self.peer_list:
                    self.disconnect_with_node(peer_id)
                    self.print_debug_messages("Peer " + peer_id + " left, disconnected.")

            self.connect_with_peers()

        if isinstance(self.privacy_protocol, AdditiveSecretSharing):
            self.privacy_protocol.update_peers_list([peer_id for peer_id in self.peer_list.keys() if peer_id != self.id])


    def quit_network(self):
        self.bulletin.remove_peer(self.id)

        self.bulletin = None
        for peer_id in list(self.connected_nodes.keys()):
                self.disconnect_with_node(peer_id)
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
            sync_time = 0.0
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

                if (i + 1) % self.sync_every != 0:
                    self.print_debug_messages("Skipping sync for iteration " + str(i))
                    continue

                sync_start_time = time.time()
                self.synchronize((epoch, i))
                sync_time += time.time() - sync_start_time

                self.print_debug_messages("Finished training iteration " + str(i))


            self.hardware_tracker.stop()

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time - sync_time
            itr_per_sec = num_batches / epoch_duration if epoch_duration > 0 else 0

            self.synchronize((epoch, "end"))

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
                self.logger.log_performance(epoch+1, self.max_epochs, len(self.peer_list), avg_train_loss, avg_val_loss, val_f1.item(), itr_per_sec, True)
                self.print_debug_messages("Early stopping triggered at epoch " + str(epoch))
                break

            self.logger.log_performance(epoch+1, self.max_epochs, len(self.peer_list), avg_train_loss, avg_val_loss, val_f1.item(), itr_per_sec, False)

        self.print_debug_messages("Finished training")


    def synchronize(self, iteration):
        # an iteration is identified by (epoch, batch index) for the syncs inside an epoch
        # and by (epoch, "end") for the sync closing the epoch
        self.iteration = iteration

        state_dict = self.model.state_dict()

        # encode weights and submit to peers
        if isinstance(self.privacy_protocol, AdditiveSecretSharing):

            self.hardware_tracker.phase_start("secret_sharing_generation")
            shared_state_dict = self.privacy_protocol.before_send(state_dict)
            self.hardware_tracker.phase_stop("secret_sharing_generation")

            for peer_id, shares in shared_state_dict.items():
                if peer_id !=  self.id:
                    self.comm_tracker.timer_start()
                    self.submit_weights(shares, True, recipient_id=peer_id)
                    self.comm_tracker.timer_stop()

            self.collect_weights(self.partial_weights_queue, self.partial_weights, iteration)

            if not self.partial_weights[iteration]:
                # nobody is at this iteration, nothing to reconstruct
                self.print_debug_messages("No peer at iteration " + str(iteration) + ", skipping sync")
                del self.partial_weights[iteration]
                return

            self.hardware_tracker.phase_start("secret_sharing_reconstruction")
            aggregated_partial_weights = self.privacy_protocol.after_receive(self.partial_weights[iteration])
            self.hardware_tracker.phase_stop("secret_sharing_reconstruction")

            cpu_aggregated_partial_weights = aggregated_partial_weights.cpu()
            self.comm_tracker.timer_start()
            self.submit_weights(cpu_aggregated_partial_weights, False)
            self.comm_tracker.timer_stop()
            del self.partial_weights[iteration]

        else:
            if self.flat_layout is None or not self.flat_layout.matches(state_dict):
                self.flat_layout = Flat(state_dict)

            # cheaper to send flat instead of the whole state_dict
            cpu_flat_weights = self.flat_layout.flatten(state_dict, device, dtype=torch.float32).cpu()
            self.comm_tracker.timer_start()
            self.submit_weights(cpu_flat_weights, False)
            self.comm_tracker.timer_stop()

        # collect weights from peers for current iteration before aggregation
        self.collect_weights(self.weights_queue, self.peers_weights, iteration)

        # update local model with aggregated weights from peers
        peers_weights = self.peers_weights.pop(iteration, {}) # clear the weights for this iteration after aggregation
        if peers_weights: # no peer at this iteration means nothing to aggregate with

            self.print_debug_messages("Updating local model with weights from iteration " + str(iteration))

            if isinstance(self.privacy_protocol, AdditiveSecretSharing):

                self.print_debug_messages("Aggregating...")
                self.hardware_tracker.phase_start("aggregation")
                aggregated_weights = self.privacy_protocol.aggregate(aggregated_partial_weights, peers_weights.values(), self.model.state_dict())
                self.hardware_tracker.phase_stop("aggregation")

            else:
                aggregated_weights = self.aggregate_weights(self.model.state_dict(), peers_weights) # hardware tacker is inside the function

            self.model.load_state_dict(aggregated_weights)


    def collect_weights(self, weights_queue, collected_weights, iteration):
        expected_weights_count = len(self.peer_list) - 1 # excluding self

        # keep weights that already arrived from peers running ahead of us
        collected_weights.setdefault(iteration, {})

        # keep track of peers that already sent a later iteration -> they are already ahead so no need to wait for them
        peers_ahead = set()

        self.comm_tracker.waiting_start()
        while len(set(collected_weights[iteration]) | peers_ahead) < expected_weights_count:
            try:
                sender_id, message = weights_queue.get(timeout=10)

                if message["iteration"] == iteration:
                    collected_weights[iteration][sender_id] = message["weights"]

                elif self._iteration_passed(message["iteration"], iteration): # avoid re-queueing
                    collected_weights.setdefault(message["iteration"], {})[sender_id] = message["weights"]
                    peers_ahead.add(sender_id)

                self.print_debug_messages("Weights from node " + str(sender_id) + " for iteration " + str(iteration) + " have been processed.")

            except queue.Empty:
                self.print_debug_messages("Timeout waiting for weights, iteration " + str(iteration))
                break

        self.comm_tracker.waiting_stop()


    def _iteration_passed(self, peer_iteration, iteration):
        # True if a peer sending peer_iteration has already left iteration behind
        peer_epoch, peer_batch = peer_iteration
        epoch, batch = iteration

        if peer_epoch != epoch:
            return peer_epoch > epoch
        if batch == "end": # the epoch sync is the last iteration of an epoch
            return False
        if peer_batch == "end":
            return True

        return peer_batch > batch


    def submit_weights(self, weights, partial_weights: bool, recipient_id=None):
        self.state = RoundState.COMMUNICATION

        message = {
            "type": "weights_submission",
            "sender_id": self.id,
            "timestamp": time.time(),
            "iteration": self.iteration,
            "batch_size": self.batch_size,
            "weights":  weights,
            "partial_weights": partial_weights,
        }

        if recipient_id is None:
            with self._lock:
                    peer_ids = list(self.connected_nodes.keys())
            for peer_id in peer_ids:
                num_bytes = self.send_to_node(peer_id, message)
                self.comm_tracker.record_sent(num_bytes)
                self.print_debug_messages("Submitted weights to node " + peer_id)
        else:
            num_bytes = self.send_to_node(recipient_id, message)
            self.comm_tracker.record_sent(num_bytes)
            self.print_debug_messages("Submitted weights to node " + recipient_id)


    def handle_weights_submission(self, node_id, data):
        self.print_debug_messages("Received weights from " + node_id)

        message = data

        if message["partial_weights"]:
            self.partial_weights_queue.put((node_id, message))
        else:
            self.weights_queue.put((node_id, message))


    def aggregate_weights(self, local_state_dict, peers_weights):
        self.print_debug_messages("Aggregating...")
        self.hardware_tracker.phase_start("aggregation")

        # aggregating flats are cheaper
        local_flat = self.flat_layout.flatten(local_state_dict, device, dtype=torch.float32)
        all_flat = [local_flat] + [w.to(device=device, dtype=torch.float32) for w in peers_weights.values()]

        mean_flat = torch.stack(all_flat).mean(dim=0)
        aggregated_weights = self.flat_layout.unflatten(mean_flat, local_state_dict)

        self.hardware_tracker.phase_stop("aggregation")
        return aggregated_weights