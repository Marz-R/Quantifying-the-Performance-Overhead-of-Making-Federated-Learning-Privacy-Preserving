import sys
from bulletin_client import BulletinClient
from peer_node import PeerNode
from toy_cnn import ToyCNN

if __name__ == "__main__":
    host       = sys.argv[1]
    port       = int(sys.argv[2])
    node_id    = sys.argv[3]
    bulletin_url = sys.argv[4] if len(sys.argv) > 4 else ""
 
    bulletin = BulletinClient(bulletin_url)
    model = ToyCNN()
    node = PeerNode(host=host, port=port, model=model, dataset="toy", id=node_id)
 
    node.start()
    node.register_to_network(bulletin)
    node.connect_with_peers()
    node.training()
    node.stop()
