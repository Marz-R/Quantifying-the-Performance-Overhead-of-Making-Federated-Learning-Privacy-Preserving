import torch
import torchvision
import torchvision.transforms as transforms
import json

def _get_dataset(dataset_name, batch_size, VALID_SPLIT, download=True):
	if dataset_name == "CIFAR10":
		# Use transforms.compose method to reformat images for modeling
		cifar10_transforms = transforms.Compose([transforms.Resize((32,32)),
												 transforms.ToTensor(),
												 transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
												 ])
		# Create Training and Testing datasets
		raw_dataset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = cifar10_transforms, download = download)
		 
	elif dataset_name == "MNIST":
		mnist_transform = transforms.Compose([transforms.ToTensor(), 
											transforms.Normalize(mean=(0.1307,), std=(0.3081,))
											])
		raw_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = mnist_transform, download = download)

	else:
		raise ValueError("Unsupported dataset")

	# Split training dataset into training and validation sets
	val_size = int(VALID_SPLIT * len(raw_dataset))
	train_size = len(raw_dataset) - val_size

	train_indices = list(range(train_size))
	val_indices = list(range(train_size, train_size + val_size))

	train_set = torch.utils.data.Subset(raw_dataset, train_indices)
	valid_set = torch.utils.data.Subset(raw_dataset, val_indices)
	
	# Instantiate loader objects to facilitate processing
	train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
	valid_loader = torch.utils.data.DataLoader(dataset = valid_set, batch_size = batch_size, shuffle = False)
	
	return train_loader, valid_loader


def load_data(dataset_name, peer_id, batch_size, VALID_SPLIT, partition_type="iid"):

	print(f"Loading {dataset_name} and {partition_type} partitioning for Peer {peer_id}...")

	train_loader, valid_loader = _get_dataset(dataset_name, batch_size, VALID_SPLIT)
	peers_indices = json.load(open(f"data/{dataset_name}_{partition_type}_partition.json", "r"))

	peer_index = peers_indices[peer_id]
	peer_subset = torch.utils.data.Subset(train_loader.dataset, peer_index)
	peer_loader = torch.utils.data.DataLoader(peer_subset, batch_size=batch_size, shuffle=True)

	return peer_loader, valid_loader