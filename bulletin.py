from dataclasses import dataclass
from abc import ABC, abstractmethod


class Observer(ABC):
    @abstractmethod
    def on_add_peer(self, peer_id: int, peer_info: dict):
        pass

    @abstractmethod
    def on_remove_peer(self, peer_id: int):
        pass


@dataclass
class Public_Bulletin:
    peer_list: dict[int, dict] # peer_id: {"host": str, "port": int}
    num_epochs: int
    _observers: list[Observer]

    # pubsub management
    def subscribe(self, observer: Observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def unsubscribe(self, observer: Observer):
        self._observers.remove(observer)

    def _notify_joined(self, peer_id: int, peer_info: dict):
        for observer in self._observers:
            observer.on_add_peer(peer_id, peer_info)

    def _notify_left(self, peer_id: int):
        for observer in self._observers:
            observer.on_remove_peer(peer_id)

    # core functions
    def add_peer(self, peer_id: int, peer_info: dict):
        if peer_id not in self.peer_list:
            self.peer_list[peer_id] = peer_info
            print("Bulletin: Peer " + peer_id + " added to bulletin.")
            self._notify_joined(peer_id, peer_info)
        else:
            print("Bulletin: Peer " + peer_id + " already joined in bulletin.")

    def remove_peer(self, peer_id: int):
        if peer_id in self.peer_list:
            del self.peer_list[peer_id]
            print("Bulletin: Peer " + peer_id + " removed from bulletin.")
            self._notify_left(peer_id)
        else:
            print("Bulletin: Peer " + peer_id + " does not exist in bulletin.")

    def get_peer_list(self):
        return self.peer_list
    
    def get_num_epochs(self):
        return self.num_epochs