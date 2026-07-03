import requests

class BulletinClient:
    def __init__(self, base_url):
        self.base_url = base_url # url of bulletin server

    def add_peer(self, peer_id, peer_info):
        requests.post(f"{self.base_url}/add_peer", json={"peer_id": peer_id, "peer_info": peer_info})

    def remove_peer(self, peer_id):
        requests.post(f"{self.base_url}/remove_peer", json={"peer_id": peer_id})

    def get_peer_list(self): # only use at registering to bulletin
        return requests.get(f"{self.base_url}/get_peer_list").json()
    
    def update_peer_list(self, last_seen_version):
        return requests.get(f"{self.base_url}/peer_list/since/{last_seen_version}").json()

    def get_num_epochs(self):
        return requests.get(f"{self.base_url}/get_num_epochs").json()["num_epochs"]