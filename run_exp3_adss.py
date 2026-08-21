#!/usr/bin/env python3
import time
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
    parser.add_argument("bulletin_ip", help="bulletin server's IP, e.g. http://192.168.1.100:5000")
    args = parser.parse_args()

    host, port, node_id = args.host, int(args.port), args.peer_id
    bulletin = BulletinClient(args.bulletin_ip)

    exp_logger = ExperimentLogger(
        peer_id=node_id, 
        model="CNN", 
        dataset="MNIST_iid", 
        privacy_protocol="AdditiveSecretSharing", 
        output_path='./measurements'
        )
    
    model = CNN(10, 1, 3*3)
    node = PeerNode(host=host, port=port, model=model, dataset="MNIST", exp_logger=exp_logger, privacy_protocol="AdditiveSecretSharing", id=node_id, sync_every=5)

    node.start()
    node.register_to_network(bulletin)
    time.sleep(10)
    node.connect_with_peers()
    node.training()
    node.quit_network()
    node.stop()