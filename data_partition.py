import torch
import json
from data_load import _get_dataset

def partition_iid(data_name, dataset, peer_list):
	# Calculate the number of samples per peer
	num_samples = len(dataset)
	num_peers = len(peer_list)
	samples_per_peer = num_samples // num_peers

	# Create a list to hold the indices for each peer
	peer_indices = []

	# Shuffle the dataset indices
	shuffled_indices = torch.randperm(num_samples)

	# Partition the shuffled indices into equal parts for each peer
	for i in range(num_peers):
		start_idx = i * samples_per_peer
		end_idx = start_idx + samples_per_peer if i < num_peers - 1 else num_samples
		peer_indices.append(shuffled_indices[start_idx:end_idx])

	_to_json(peer_list, peer_indices, dataset_name=data_name, partition_type="iid")


def partition_non_iid():
	pass


def _to_json(peer_list, peer_indices, dataset_name, partition_type):
    with open(f"data/{dataset_name}_{partition_type}_partition.json", "w") as f:
        json.dump({peer_id: indices.tolist() for peer_id, indices in zip(peer_list, peer_indices)}, f)


if __name__ == "__main__":
    dataset_name = "MNIST"
    peer_list = ["03", "04", "05"]
    BATCH_SIZE = 128
    VALID_SPLIT = 0.2

    train_loader, _ = _get_dataset(dataset_name, BATCH_SIZE, VALID_SPLIT)
    partition_iid(dataset_name, train_loader.dataset, peer_list)