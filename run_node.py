import argparse
from bulletin_client import BulletinClient
from model_cnn import CNN
from peer_node import PeerNode
from measurements.exp_logger import ExperimentLogger

# input: host port ID bulletinIP model dataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host")
    parser.add_argument("port")
    parser.add_argument("peer_id", help="integer")
    parser.add_argument("bulletin_ip", help="bulletin server's IP, e.g. http://192.168.1.100:5000")
    parser.add_argument("model", help="which model to use: resnet18 or cnn")
    parser.add_argument("dataset", help="which dataset to use: cifar10, cifar100 or mnist")
    parser.add_argument("batch_size", help="default 128")
    parser.add_argument("num_nodes", help="this is just for exp_logger reference")
    parser.add_argument("max_epoch", help="this is just for exp_logger reference, to adjust max_epoch, go to bulletin_server")
    parser.add_argument("privacy_protocol")
    parser.add_argument("sync_every", help="how often are the peers exchanging weights, default 5 iterations")
    args = parser.parse_args()

    host, port, node_id = args.host, int(args.port), args.peer_id # input IP and ID
    bulletin = BulletinClient(args.bulletin_ip)

    exp_logger = ExperimentLogger()

    model = CNN(10, 1, 3*3)
    node = PeerNode(host=host, port=port, model=args.model, dataset=args.dataset, exp_logger=exp_logger, id=node_id, sync_every=args.sync_every)

    node.start()
    node.register_to_network(bulletin)
    node.connect_with_peers()
    node.training()
    node.quit_network()
    node.stop()