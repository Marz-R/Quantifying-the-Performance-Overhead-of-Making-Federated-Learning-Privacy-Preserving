import torch
import json
import argparse
from data_load import _get_dataset

def partition_iid(data_name, dataset, peer_list):
	# Calculate the number of samples per peer
	num_samples = len(dataset)
	num_peers = len(peer_list)
	samples_per_peer = num_samples // num_peers

	peer_indices = [[] for _ in range(num_peers)]

	# Shuffle the dataset indices
	shuffled_indices = torch.randperm(num_samples)

	# Partition the shuffled indices into equal parts for each peer
	for i in range(num_peers):
		start_idx = i * samples_per_peer
		end_idx = start_idx + samples_per_peer if i < num_peers - 1 else num_samples
		peer_indices[i].append(shuffled_indices[start_idx:end_idx])

	# Glue the slices of every peer back together
	peer_indices = [torch.cat(indices) for indices in peer_indices]

	_to_json(peer_list, peer_indices, dataset_name=data_name, partition_type="iid")


def partition_non_iid(data_name, dataset, peer_list, classes_per_peer=None):
	# Get the label of every sample in the dataset
	labels = _get_labels(dataset)
	classes = labels.unique().tolist()
	num_peers = len(peer_list)
	num_classes = len(classes)

	# default half of all classes per peer
	if classes_per_peer is None:
		classes_per_peer = num_classes // 2

	class_holders = _deal_classes(classes, num_peers, classes_per_peer)

	peer_indices = [[] for _ in range(num_peers)]

	# Split every class among the peers that hold it
	for label, holders in class_holders.items():
		class_indices = (labels == label).nonzero(as_tuple=True)[0]
		num_class_samples = len(class_indices)

		# Uneven shares by holder's position in list not id
		weights = torch.tensor([((label + i) % num_peers) + 1 for i in holders], dtype=torch.float)

		# Turn the weights into cutting points inside the class
		boundaries = (weights.cumsum(0) / weights.sum() * num_class_samples).round().long()

		# Hand each peer its slice of the class
		start_idx = 0
		for j, i in enumerate(holders):
			end_idx = boundaries[j].item()
			peer_indices[i].append(class_indices[start_idx:end_idx])
			start_idx = end_idx

	peer_indices = [torch.cat(indices) for indices in peer_indices]

	_report(peer_list, peer_indices, labels, partition_type="non_iid")
	_to_json(peer_list, peer_indices, dataset_name=data_name, partition_type="non_iid")


def _deal_classes(classes, num_peers, classes_per_peer) -> dict:
	num_classes = len(classes)
	class_holders = {label: [] for label in classes}

	# round-robin assignment of classes to peers
	for i in range(num_peers):
		start_class = round(i * num_classes / num_peers)
		for j in range(classes_per_peer):
			label = classes[(start_class + j) % num_classes]
			if i not in class_holders[label]:
				class_holders[label].append(i)

	# All classes must have at least one holder
	for label, holders in class_holders.items():
		if not holders:
			raise ValueError(f"Class {label} ended up on no peer, raise classes_per_peer")

	return class_holders


def _get_labels(dataset):
	# Follow the subsets down to the raw dataset to get the labels
	if isinstance(dataset, torch.utils.data.Subset):
		labels = _get_labels(dataset.dataset)
		return labels[torch.tensor(dataset.indices, dtype=torch.long)]

	if hasattr(dataset, "targets"):
		targets = dataset.targets
	else:
		targets = [label for _, label in dataset] # just in case

	return torch.as_tensor(targets, dtype=torch.long)


def _report(peer_list, peer_indices, labels, partition_type):
	classes = labels.unique().tolist()

	print(f"\n{partition_type} partition")
	print("peer  " + "".join(f"{c:>7}" for c in classes) + f"{'total':>9}")

	for peer_id, indices in zip(peer_list, peer_indices):
		peer_labels = labels[indices]
		class_counts = "".join(f"{int((peer_labels == c).sum()):>7}" for c in classes)
		print(f"{peer_id:<6}{class_counts}{len(indices):>9}")
	print()


def _to_json(peer_list, peer_indices, dataset_name, partition_type):
	with open(f"data/{dataset_name}_{partition_type}_partition.json", "w") as f:
		json.dump({peer_id: indices.tolist() for peer_id, indices in zip(peer_list, peer_indices)}, f)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=str, default="MNIST", help="Dataset name (default: MNIST)")
	parser.add_argument("--partition", type=str, default="iid", choices=["iid", "non_iid"],
						help="Partition scheme (default: iid)")
	parser.add_argument("--peers", type=str, nargs="+", default=["03", "04", "05"],
						help="Peer ids (default: 03 04 05)")
	parser.add_argument("--classes-per-peer", type=int, default=None,
						help="How many classes a peer holds ***non iid only*** (default: half of the classes)")
	args = parser.parse_args()

	dataset_name = args.dataset
	peer_list = args.peers
	BATCH_SIZE = 128
	VALID_SPLIT = 0.2

	train_loader, _ = _get_dataset(dataset_name, BATCH_SIZE, VALID_SPLIT)

	if args.partition == "iid":
		partition_iid(dataset_name, train_loader.dataset, peer_list)
	else:
		partition_non_iid(dataset_name, train_loader.dataset, peer_list, args.classes_per_peer)
