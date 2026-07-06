import argparse
from bulletin_client import BulletinClient
from model_cnn import CNN
from peer_node import PeerNode

# input: host port ID bulletinIP model dataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host")
    parser.add_argument("port")
    parser.add_argument("peer_id")
    parser.add_argument("bulletin_ip")
    #parser.add_argument("model", help="which ML model to use: resnet18 or cnn")
    #parser.add_argument("dataset", help="which dataset to use: cifar10, cifar100 or mnist")
    args = parser.parse_args()

    host, port, node_id = args.host, int(args.port), args.peer_id # input IP and ID
    bulletin = BulletinClient(args.bulletin_ip)  # bulletin server's IP, e.g. http://192.168.1.100:5000

    model = CNN(10, 1, 3*3)
    node = PeerNode(host=host, port=port, model=model, dataset="MNIST", id=node_id)

    node.start()
    node.register_to_network(bulletin)
    node.connect_with_peers()
    node.training()
    node.stop