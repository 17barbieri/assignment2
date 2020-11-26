import os
import pdb
import random
import numpy as np
import pandas as pd

from PIL import Image
from zipfile import ZipFile

import torch
import torchvision
import torchvision.transforms as transforms

from datasets.dataset_utils import download_MNIST, download_CIFAR100, download_Omniglot, sample_class_labels, get_simple_loader


def load_datasets(dataset, batch_size, data_percent, n_classes = None, train_classes = False):
	if dataset == 'MNIST':
		N = load_MNIST_dataset([0],1).number_of_classes() # Retrieve total number of classes (here 10)
		classes_to_load = np.arange(0, N)
		if n_classes is not None: # Define dataset for one-shot
			random.shuffle(classes_to_load)
			if train_classes == True: # If training process
				classes_to_load = classes_to_load[:n_classes] # Select the n_classes first classes for training
			else:
				classes_to_load = classes_to_load[n_classes:] # Select the (N-n_classes) last classes for one-shot testing
		else: # Normal training
			pass

		train_dataset = load_MNIST_dataset(classes_to_load, data_percent, train = True)
		train_loader = get_simple_loader(train_dataset, batch_size)

		test_dataset = load_MNIST_dataset(classes_to_load, data_percent, train = False)
		test_loader = get_simple_loader(test_dataset, batch_size)

	elif dataset == 'CIFAR100':
		N = load_CIFAR100_dataset([0],1).number_of_classes() # Retrieve total number of classes, here 100
		classes_to_load = np.arange(0, N)
		if n_classes is not None: # Define dataset for one-shot
			random.shuffle(classes_to_load)
			if train_classes == True: # If training process
				classes_to_load = classes_to_load[:n_classes] # Select the n_classes first classes for training
			else:
				classes_to_load = classes_to_load[n_classes:] # Select the (N-n_classes) last classes for one-shot testing
		else: # Normal training
			pass

		train_dataset = load_CIFAR100_dataset(classes_to_load, data_percent, train = True)
		train_loader = get_simple_loader(train_dataset, batch_size)

		test_dataset = load_CIFAR100_dataset(classes_to_load, data_percent, train = False)
		test_loader = get_simple_loader(test_dataset, batch_size)

	elif dataset == 'Omniglot':
		N = load_MNIST_dataset([0],1).number_of_classes()
		pdb.set_trace()
		classes_to_load = np.arange(0, N)
		if n_classes is not None: # Define dataset for one-shot
			random.shuffle(classes_to_load)
			if train_classes == True: # If training process
				classes_to_load = classes_to_load[:n_classes] # Select the n_classes first classes for training
			else:
				classes_to_load = classes_to_load[n_classes:] # Select the (N-n_classes) last classes for one-shot testing
		else: # Normal training
			pass
		
		train_dataset = load_Omniglot_dataset(classes_to_load, data_percent, train = True)
		train_loader = get_simple_loader(train_dataset, batch_size)

		test_dataset = load_Omniglot_dataset(classes_to_load, data_percent, train = False)
		test_loader = get_simple_loader(test_dataset, batch_size)

	else:
		print('The selected dataset has not yet been implemented')
		raise NotImplementedError
	
	return train_loader, test_loader

class load_MNIST_dataset:
	def __init__(self, classes_to_load, data_percentage, train = True):
		self.classes_to_load = classes_to_load
		self.data_percentage = data_percentage
		
		if os.path.isdir('./data/MNIST/pt_sets') == False:
			if os.path.isfile('./data/MNIST/pt_sets.zip'):
				zip_name = './data/MNIST/pt_sets.zip'
				with ZipFile(zip_name, 'r') as zip:
					zip.extractall('./data/MNIST')
			else:
				download_MNIST()
				print('Missing zip file')
		
		if train == True:
			self.data_initial, self.labels_initial = torch.load('./data/MNIST/pt_sets/train_set.pt')
		if train == False:
			self.data_initial, self.labels_initial = torch.load('./data/MNIST/pt_sets/test_set.pt')
		
		# Randomly selecting x% of the elements of each class to load 
		self.cls_indexes=sample_class_labels(self.classes_to_load, self.labels_initial, self.data_percentage)

		self.data = self.data_initial[np.concatenate(self.cls_indexes)].float()
		self.labels = self.labels_initial[np.concatenate(self.cls_indexes)].long()

	def __len__(self):
		return len(self.data)

	def number_of_classes(self):
		return len(set(self.labels_initial.detach().numpy())) # Return the total number of clsses of the dataset

	def __getitem__(self, idx):
		data = self.data[idx]
		label = self.labels[idx].item() # Original label in [0, N]
		label_1 = np.where(self.classes_to_load == label)
		label_2 = torch.tensor(int(label_1[0])) # Label in [0, n_classes] (or [N-n_classes, N])
		return data, label_2

class load_CIFAR100_dataset:
	def __init__(self, classes_to_load, data_percentage, train = True):
		self.classes_to_load = classes_to_load
		self.data_percentage = data_percentage
		if os.path.isdir('./data/CIFAR100/pt_sets') == False:
			if os.path.isfile('./data/CIFAR100/pt_sets.zip'):
				zip_name = './data/CIFAR100/pt_sets.zip'
				with ZipFile(zip_name, 'r') as zip:
					zip.extractall('./data/CIFAR100')
			else:
				download_CIFAR100()
				print('Missing zip file')
		
		if train == True:
			self.data_initial, self.labels_initial = torch.load('./data/CIFAR100/pt_sets/train_set.pt')
		if train == False:
			self.data_initial, self.labels_initial = torch.load('./data/CIFAR100/pt_sets/test_set.pt')
		
		# Randomly selecting x% of the elements of each class to load 
		self.cls_indexes=sample_class_labels(self.classes_to_load, self.labels_initial, self.data_percentage)

		self.data = self.data_initial[np.concatenate(self.cls_indexes)].float()
		self.labels = self.labels_initial[np.concatenate(self.cls_indexes)].long()

	def __len__(self):
		return len(self.data)

	def number_of_classes(self):
		return len(set(self.labels_initial.detach().numpy())) # Return the total number of clsses of the dataset

	def __getitem__(self, idx):
		data = self.data[idx]
		label = self.labels[idx].item() # Original label in [0, N]
		label_1 = np.where(self.classes_to_load == label)
		label_2 = torch.tensor(int(label_1[0])) # Label in [0, n_classes] (or [N-n_classes, N])
		return data, label_2

class load_Omniglot_dataset:
	def __init__(self, classes_to_load, data_percentage, train = True):
		download_Omniglot()
		self.classes_to_load = classes_to_load
		self.data_percentage = data_percentage
		if train == True:
			self.data_initial, self.labels_initial = torch.load('./data/omniglot-py/organized_data/pt_sets/train_set.pt')
		if train == False:
			self.data_initial, self.labels_initial = torch.load('./data/omniglot-py/organized_data/pt_sets/test_set.pt')

		self.cls_indexes=sample_class_labels(self.classes_to_load, self.labels_initial, self.data_percentage)

		self.data = self.data_initial[np.concatenate(self.cls_indexes)].float()
		self.data = self.data.view(self.data.size(0), 1, self.data.size(1), self.data.size(2))
		self.labels = self.labels_initial[np.concatenate(self.cls_indexes)].long()

	def __len__(self):
		return len(self.data)

	def number_of_classes(self):
		return len(set(self.labels_initial.detach().numpy())) # Return the total number of clsses of the dataset

	def __getitem__(self, idx):
		data = self.data[idx]
		label = self.labels[idx].item() # Original label in [0, N]
		label_1 = np.where(self.classes_to_load == label)
		label_2 = torch.tensor(int(label_1[0])) # Label in [0, n_classes] (or [N-n_classes, N])
		return data, label_2