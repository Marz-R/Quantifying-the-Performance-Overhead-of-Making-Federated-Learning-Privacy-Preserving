import threading
from peer_node import PeerNode
from bulletin import Public_Bulletin
from model_cnn import CNN

# Hyperparameters
NUM_EPOCHS = 50

def test_peer_node():
    model = CNN(10, 1, 3*3)
    bulletin = Public_Bulletin({}, NUM_EPOCHS, [])

    node1 = PeerNode(host="localhost", port=8000, model=model, dataset="MNIST", id=10)
    node2 = PeerNode(host="localhost", port=8001, model=model, dataset="MNIST", id=20)
    node3 = PeerNode(host="localhost", port=8002, model=model, dataset="MNIST", id=30)

    peers = [node1, node2, node3]

     # run each phase concurrently across all nodes
    for func in ["start", "register_to_network", "connect_with_peers", "training"]:
        threads = []
        for node in peers:
            method = getattr(node, func)
            args = (bulletin,) if func == "register_to_network" else ()
            t = threading.Thread(target=method, args=args)
            threads.append(t)
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()  # Wait for all nodes to finish this phase before moving on

    node1.stop()
    node2.stop()
    node3.stop()


if __name__ == "__main__":
    test_peer_node()