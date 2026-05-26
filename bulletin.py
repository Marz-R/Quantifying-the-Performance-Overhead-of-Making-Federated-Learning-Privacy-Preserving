from dataclasses import dataclass

@dataclass
class Public_Bulletin:
    peer_list: dict[int, dict] # peer_id: {"host": str, "port": int}
    num_epochs: int
    def add_peer(self, peer_id: int, peer_info: dict):
        self.peer_list[peer_id] = peer_info
        print(f"Peer {peer_id} added to bulletin.")

    def remove_peer(self, peer_id: int):
        if peer_id in self.peer_list:
            del self.peer_list[peer_id]
            print(f"Peer {peer_id} removed from bulletin.")

    def get_peer_list(self):
        return self.peer_list
    
    def get_num_epochs(self):
        return self.num_epochs