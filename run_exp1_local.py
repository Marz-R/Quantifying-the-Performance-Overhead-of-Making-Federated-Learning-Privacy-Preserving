import argparse
from bulletin_client import BulletinClient
from model_cnn import CNN
from peer_node import PeerNode
from measurements.exp_logger import ExperimentLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host")
    parser.add_argument("port")
    parser.add_argument("peer_id", help="integer")
    parser.add_argument("bulletin_ip", help="in this case, it should be localhost but different port")
    args = parser.parse_args()

    host, port, node_id = args.host, int(args.port), args.peer_id
    bulletin = BulletinClient(args.bulletin_ip)

    exp_logger = ExperimentLogger(
        peer_id=node_id, 
        model="CNN", 
        dataset="MNIST", 
        privacy_protocol="LocalML", 
        output_path='./measurements'
        )
    
    model = CNN(10, 1, 3*3)
    node = PeerNode(host=host, port=port, model=model, dataset="MNIST", exp_logger=exp_logger, id=node_id, sync_every=10)

    node.start()
    node.register_to_network(bulletin)
    node.training()
    node.quit_network()
    node.stop()