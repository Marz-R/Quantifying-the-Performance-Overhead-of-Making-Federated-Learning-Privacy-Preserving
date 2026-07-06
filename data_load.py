import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split


def load_dataset(dataset_name, BATCH_SIZE, VALID_SPLIT, download):

	if dataset_name == "CIFAR10":
		return load_cifar10(BATCH_SIZE, VALID_SPLIT, download)
	elif dataset_name == "CIFAR100":
		return load_cifar100(BATCH_SIZE, VALID_SPLIT, download)
	elif dataset_name == "MNIST":
		return load_mnist(BATCH_SIZE, VALID_SPLIT, download)
	elif dataset_name == "toy":
		return load_toy_mnist()
	else:
		raise ValueError("Unsupported dataset")


def load_toy_mnist(batch_size=32, val_split=0.2, num_samples=1000):

	print("Loading toy dataset...")

	transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
 
	full_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=False, transform=transform)
 
	subset = torch.utils.data.Subset(full_dataset, indices=range(num_samples))
 
	val_size   = int(num_samples * val_split)
	train_size = num_samples - val_size
	train_data, val_data = random_split(subset, [train_size, val_size])
 
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
	val_loader   = torch.utils.data.DataLoader(val_data,   batch_size=batch_size, shuffle=False)
	test_loader = None
 
	return train_loader, val_loader, test_loader


def load_cifar10(BATCH_SIZE, VALID_SPLIT, download):

	print("Loading CIFAR10 dataset...")

	# Use transforms.compose method to reformat images for modeling
	cifar10_transforms = transforms.Compose([transforms.Resize((32,32)),
	                                     transforms.ToTensor(),
	                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
	                                     ])

	# Create Training and Testing datasets
	train_cifar10_dataset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = cifar10_transforms, download = download)
	test_cifar10_dataset = torchvision.datasets.CIFAR10(root = './data', train = False, transform = cifar10_transforms, download = download)
 
	# Split training dataset into training and validation sets
	val_size = int(VALID_SPLIT * len(train_cifar10_dataset))
	train_size = len(train_cifar10_dataset) - val_size
	train_cifar10_dataset, valid_cifar10_dataset = random_split(train_cifar10_dataset, [train_size, val_size])

	# Instantiate loader objects to facilitate processing
	train_cifar10_loader = torch.utils.data.DataLoader(dataset = train_cifar10_dataset, batch_size = BATCH_SIZE, shuffle = True)
	valid_cifar10_loader = torch.utils.data.DataLoader(dataset = valid_cifar10_dataset, batch_size = BATCH_SIZE, shuffle = False)
	test_cifar10_loader = torch.utils.data.DataLoader(dataset = test_cifar10_dataset, batch_size = BATCH_SIZE, shuffle = True)

	return train_cifar10_loader, valid_cifar10_loader, test_cifar10_loader


def load_cifar100(BATCH_SIZE, VALID_SPLIT, download):

	print("Loading CIFAR100 dataset...")

	cifar100_transforms = transforms.Compose([transforms.Resize((32,32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
                                     ])

	train_cifar100_dataset = torchvision.datasets.CIFAR100(root = './data', train = True, transform = cifar100_transforms, download = download)
	test_cifar100_dataset = torchvision.datasets.CIFAR100(root = './data', train = False, transform = cifar100_transforms, download = download)

	val_size = int(VALID_SPLIT * len(train_cifar100_dataset))
	train_size = len(train_cifar100_dataset) - val_size
	train_cifar100_dataset, valid_cifar100_dataset = random_split(train_cifar100_dataset, [train_size, val_size])

	train_cifar100_loader = torch.utils.data.DataLoader(dataset = train_cifar100_dataset, batch_size = BATCH_SIZE, shuffle = True)
	valid_cifar100_loader = torch.utils.data.DataLoader(dataset = valid_cifar100_dataset, batch_size = BATCH_SIZE, shuffle = False)
	test_cifar100_loader = torch.utils.data.DataLoader(dataset = test_cifar100_dataset, batch_size = BATCH_SIZE, shuffle = True)

	return train_cifar100_loader, valid_cifar100_loader, test_cifar100_loader


def load_mnist(BATCH_SIZE, VALID_SPLIT, download):

	print("Loading MNIST dataset...")

	mnist_transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
                                    ])

	train_mnist_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = mnist_transform, download = download)
	test_mnist_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = mnist_transform, download = download)

	val_size = int(VALID_SPLIT * len(train_mnist_dataset))
	train_size = len(train_mnist_dataset) - val_size
	train_mnist_dataset, valid_mnist_dataset = random_split(train_mnist_dataset, [train_size, val_size])

	train_mnist_loader = torch.utils.data.DataLoader(dataset = train_mnist_dataset, batch_size = BATCH_SIZE, shuffle = True)
	valid_mnist_loader = torch.utils.data.DataLoader(dataset = valid_mnist_dataset, batch_size = BATCH_SIZE, shuffle = False)
	test_mnist_loader = torch.utils.data.DataLoader(dataset = test_mnist_dataset, batch_size = BATCH_SIZE, shuffle = True)

	return train_mnist_loader, valid_mnist_loader, test_mnist_loader

