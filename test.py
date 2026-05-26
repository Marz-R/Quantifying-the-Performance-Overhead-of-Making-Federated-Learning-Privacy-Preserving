from peer_node import PeerNode
from bulletin import Public_Bulletin
from model_cnn import CNN
import time

def test_peer_node():
    model = CNN(10, 1, 3*3)
    bulletin = Public_Bulletin({}, 20)
    node1 = PeerNode(host="localhost", port=8000, model=model, dataset="MNIST", id=10, bulletin=bulletin)
    node2 = PeerNode(host="localhost", port=8001, model=model, dataset="MNIST", id=20, bulletin=bulletin)
    node3 = PeerNode(host="localhost", port=8002, model=model, dataset="MNIST", id=30, bulletin=bulletin)

    node1.start()
    node1.debug = False
    node2.start()
    node2.debug = False
    node3.start()
    node3.debug = False
    time.sleep(1)

    node1.register_to_network()
    time.sleep(1)
    node2.register_to_network()
    time.sleep(1)
    node3.register_to_network()
    time.sleep(1)

    node1.connect_with_peers()
    time.sleep(2)
    node2.connect_with_peers()
    time.sleep(2)
    node3.connect_with_peers()
    time.sleep(2)

    #node1.send_to_nodes('{"message": "hoi from node 1", "sender_id": 1, "type": "test_message"}')

    node1.training()

    time.sleep(100)

    node1.stop()
    node2.stop()
    node3.stop()


if __name__ == "__main__":
    test_peer_node()