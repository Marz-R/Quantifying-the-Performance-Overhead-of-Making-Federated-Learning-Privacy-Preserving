from dataclasses import dataclass

@dataclass
class Public_Bulletin:
    peer_list: dict[int, dict]  # peer_id: {"host": str, "port": int}
    num_epochs: int             # max number of epochs for training
    version: int = 0 # version number of peer_list

    def add_peer(self, peer_id: int, peer_info: dict):
        if peer_id not in self.peer_list:
            self.peer_list[peer_id] = peer_info
            self.version += 1
            print("Bulletin: Peer " + peer_id + " added to bulletin.")
            self._notify_joined(peer_id, peer_info)
        else:
            print("Bulletin: Peer " + peer_id + " already joined in bulletin.")

    def remove_peer(self, peer_id: int):
        if peer_id in self.peer_list:
            del self.peer_list[peer_id]
            self.version += 1
            print("Bulletin: Peer " + peer_id + " removed from bulletin.")
            self._notify_left(peer_id)
        else:
            print("Bulletin: Peer " + peer_id + " does not exist in bulletin.")

    def get_peer_list(self):
        return self.peer_list
    
    def get_num_epochs(self):
        return self.num_epochs