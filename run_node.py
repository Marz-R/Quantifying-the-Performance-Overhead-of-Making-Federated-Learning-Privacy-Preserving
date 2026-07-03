# run_node.py — same file run on every machine, with different args
import sys
from bulletin_client import BulletinClient
from model_cnn import CNN
from peer_node import PeerNode

if __name__ == "__main__":
    host, port, node_id = sys.argv[1], int(sys.argv[2]), sys.argv[3] # input IP and ID
    model = CNN(10, 1, 3*3)
    bulletin = BulletinClient("")  # bulletin server's IP, e.g. http://192.168.1.100:5000

    node = PeerNode(host=host, port=port, model=model, dataset="MNIST", id=node_id)
    node.start()
    node.register_to_network(bulletin)
    node.connect_with_peers()
    node.training()
    node.stop