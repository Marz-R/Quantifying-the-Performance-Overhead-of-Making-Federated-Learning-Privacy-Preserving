#!/usr/bin/env python3
"""
    # Local ML baseline (single machine, no peer networking)
    ./run_exp.py localhost 5001 0 localhost:6000 -p local

    # Plaintext FL
    ./run_exp.py 192.168.1.10 5001 0 http://192.168.1.100:5000 -p plaintext

    # Additive secret sharing, overriding dataset
    ./run_exp.py 192.168.1.10 5001 0 http://192.168.1.100:5000 -p adss -d CIFAR10
"""
import argparse
import time

from bulletin_client import BulletinClient
from model_cnn import CNN
from peer_node import PeerNode
from measurements.exp_logger import ExperimentLogger

# Per-protocol defaults, mirroring the three original scripts.
PROTOCOL_DEFAULTS = {
    "local": {
        "privacy_protocol": "LocalML",
        "model": "CNN",
        "dataset": "MNIST",
        "partition": None, 
        "needs_peer_networking": False,
    },
    "plaintext": {
        "privacy_protocol": "Plaintext",
        "model": "CNN",
        "dataset": "MNIST",
        "partition": "iid",
        "needs_peer_networking": True,
    },
    "adss": {
        "privacy_protocol": "AdditiveSecretSharing",
        "model": "CNN",
        "dataset": "MNIST",
        "partition": "iid",
        "needs_peer_networking": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a P2PFL experiment (local / plaintext / additive secret sharing)."
    )
    parser.add_argument("host")
    parser.add_argument("port")
    parser.add_argument("peer_id", help="integer")
    parser.add_argument(
        "bulletin_ip",
        help="Bulletin server address, e.g. http://192.168.1.100:5000 "
             "(for --protocol local, localhost on a different port is fine)",
    )
    parser.add_argument(
        "-p", "--protocol",
        choices=PROTOCOL_DEFAULTS.keys(),
        required=True,
        help="Which experiment configuration to run.",
    )

    # Optional overrides for the per-protocol defaults above.
    parser.add_argument("-m", "--model", type=str, default="CNN",
                         help="Override the default model to train.")
    parser.add_argument("-d", "--dataset", type=str, default="MNIST",
                         help="Override the dataset name used for training.")
    parser.add_argument("--partition", type=str, default=None, choices=["iid", "non_iid"],
                         help="Override how the dataset is partitioned across the peers.")

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = PROTOCOL_DEFAULTS[args.protocol]

    host, port, node_id = args.host, int(args.port), args.peer_id
    bulletin = BulletinClient(args.bulletin_ip)

    chosen_protocol = args.protocol 
    chosen_model = args.model
    chosen_dataset = args.dataset

    exp_logger = ExperimentLogger(
        peer_id=node_id,
        model=chosen_model,
        dataset=chosen_dataset,
        privacy_protocol=chosen_protocol,
        output_path="./measurements",
    )

    if chosen_model == "CNN" and chosen_dataset == "MNIST":
        model = CNN(10, 1, 3*3) 
    elif chosen_model == "CNN" and chosen_dataset == "CIFAR10":
        model = CNN(10, 3, 4*4) 

    chosen_partition = args.partition or cfg["partition"] or "iid"

    peer_node_kwargs = dict(
        host=host,
        port=port,
        model=model,
        dataset=chosen_dataset,
        exp_logger=exp_logger,
        id=node_id,
        sync_every=5,
        partition_type=chosen_partition
    )

    if cfg["privacy_protocol"] == "AdditiveSecretSharing":
        peer_node_kwargs["privacy_protocol"] = cfg["privacy_protocol"]
    else:
        peer_node_kwargs["privacy_protocol"] = None

    node = PeerNode(**peer_node_kwargs)

    node.start()
    node.register_to_network(bulletin)

    if cfg["needs_peer_networking"]:
        time.sleep(10)
        node.connect_with_peers()

    node.training()
    node.quit_network()
    node.stop()


if __name__ == "__main__":
    main()