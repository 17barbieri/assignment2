import os
import pdb
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision
from torch.utils.data import DataLoader
from zipfile import ZipFile
from shutil import copyfile
from random import sample

def sample_class_labels(classes_to_load, labels_initial, data_percentage):
	cls_indexes = []
	for j in classes_to_load:
		class_indexes = (labels_initial==j).nonzero()
		class_indexes = sample(class_indexes.view(-1).tolist(), int(len(class_indexes)*data_percentage))
		cls_indexes.append(class_indexes)
	return cls_indexes

def get_simple_loader(dataset, batch_size=1):
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	kwargs = {'num_workers': 4, 'pin_memory': False} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, shuffle = True, **kwargs)
	return loader

def download_MNIST(batch_size_train = 100, batch_size_test = 100):
	print('Downloading and preprocessing MNIST')
	os.makedirs('data', exist_ok = True)
	train_loader = torch.utils.data.DataLoader(
		torchvision.datasets.MNIST('./data', train=True, download=True,
								transform=torchvision.transforms.Compose([
								torchvision.transforms.ToTensor(),
								torchvision.transforms.Normalize(
								(0.1307,), (0.3081,))
								])),
		batch_size=batch_size_train, shuffle=True)

	test_loader = torch.utils.data.DataLoader(
		torchvision.datasets.MNIST('./data', train=False, download=True,
								transform=torchvision.transforms.Compose([
								torchvision.transforms.ToTensor(),
								torchvision.transforms.Normalize(
								(0.1307,), (0.3081,))
								])),
		batch_size=batch_size_test, shuffle=True)

	print('Creating train data')
	train_data, train_labels = create_pt_file(train_loader)
	print('Creating test data')
	test_data, test_labels = create_pt_file(test_loader)
	
	# Safing preprocessed data
	train_set = [train_data, train_labels]
	test_set = [test_data, test_labels]
	save_path = './data/MNIST/pt_sets/'
	os.makedirs(save_path, exist_ok = True)
	torch.save(train_set, save_path + 'train_set.pt')
	torch.save(test_set, save_path + 'test_set.pt')
	with open(save_path + '.gitignore', 'w') as f:
		print('train_set.pt', file = f)
		print('test_set.pt', file = f)
	f.close
	with open('./data/MNIST/.gitignore', 'w') as f:
		print('pt_sets', file = f)
	f.close
	with ZipFile('./data/MNIST/pt_sets.zip', 'w') as zipObj:
		zipObj.write('./data/MNIST/pt_sets/train_set.pt')
		zipObj.write('./data/MNIST/pt_sets/test_set.pt')
	zipObj.close()

def download_CIFAR100(batch_size_train = 100, batch_size_test = 100):
	print('Downloading and preprocessing MNIST')
	os.makedirs('data', exist_ok = True)
	train_loader = torch.utils.data.DataLoader(
		torchvision.datasets.CIFAR100(root = './data',
									  train = True,
									  transform = torchvision.transforms.Compose([
													torchvision.transforms.ToTensor(),
													torchvision.transforms.Normalize(
													(0.1307,), (0.3081,))
													]),
									  download = True),
			batch_size=batch_size_test, shuffle=True)
	test_loader = torch.utils.data.DataLoader(
		torchvision.datasets.CIFAR100(root = './data',
									  train = False,
									  transform = torchvision.transforms.Compose([
													torchvision.transforms.ToTensor(),
													torchvision.transforms.Normalize(
													(0.1307,), (0.3081,))
													]),
									  download = True),
			batch_size=batch_size_test, shuffle=True)
	print('Creating train data')
	train_data, train_labels = create_pt_file(train_loader)
	print('Creating test data')
	test_data, test_labels = create_pt_file(test_loader)
	
	# Safing preprocessed data
	train_set = [train_data, train_labels]
	test_set = [test_data, test_labels]
	save_path = './data/CIFAR100/pt_sets/'
	os.makedirs(save_path, exist_ok = True)
	torch.save(train_set, save_path + 'train_set.pt')
	torch.save(test_set, save_path + 'test_set.pt')
	with open(save_path + '.gitignore', 'w') as f:
		print('train_set.pt', file = f)
		print('test_set.pt', file = f)
	f.close
	with open('./data/CIFAR100/.gitignore', 'w') as f:
		print('pt_sets', file = f)
		print('pt_sets.zip', file = f)
	f.close
	with ZipFile('./data/CIFAR100/pt_sets.zip', 'w') as zipObj:
		zipObj.write('./data/CIFAR100/pt_sets/train_set.pt')
		zipObj.write('./data/CIFAR100/pt_sets/test_set.pt')
	zipObj.close()


def download_Omniglot():
	if os.path.isdir('./data/omniglot-py/images_background') == False or os.path.isdir('./data/omniglot-py/images_evaluation') == False:
		train_dataset = torchvision.datasets.Omniglot(root="./data",
								 					  download=True,
								 					  background = True,
								 					  transform=torchvision.transforms.ToTensor()
								 					  )
		test_dataset = torchvision.datasets.Omniglot(root="./data",
													 download=True,
													 background = False,
													 transform=torchvision.transforms.ToTensor()
													 )
	#Organizing data
	data_dir = './data/omniglot-py'
	transform = torchvision.transforms.Compose([
								torchvision.transforms.ToTensor()
								])
	for data_type in ['images_background', 'images_evaluation']:
		organized_dir = './data/omniglot-py/organized_data'
		if data_type == 'images_background':
			sub_organized_dir = os.path.join(organized_dir, 'train')
		else:
			sub_organized_dir = os.path.join(organized_dir, 'test')
		zip_name = os.path.join(organized_dir, 'pt_sets.zip')
		if os.path.isfile(os.path.join(organized_dir, 'pt_sets/train_set.pt')) == True and os.path.isfile(os.path.join(organized_dir, 'pt_sets/test_set.pt')) == True :
			break
		if os.path.isfile(zip_name):
			print('Extracting zip archive')
			with ZipFile(zip_name, 'r') as zip:
				zip.extractall(organized_dir)
			break
		print('Creation of the dataset pt files, this will run only once.')
		os.makedirs(sub_organized_dir, exist_ok = True)
		current_label = 0
		current_id = 0
		image_names = []
		images_torch = None
		image_ids = []
		image_original_labels = []
		labels = []
		image_paths = []
		print('Preprocessing {}:'.format(data_type))
		alphabets = sorted(os.listdir(os.path.join(data_dir, data_type)))
		for alphabet in alphabets:
			alphabet_path = os.path.join(data_dir, data_type, alphabet)
			characters = sorted(os.listdir(alphabet_path))
			for character in characters:	
				character_path = os.path.join(alphabet_path, character)
				images = sorted(os.listdir(character_path))
				for image in images:
					image_names.append(image)
					image_ids.append(current_id)
					labels.append(current_label)
					original_label = alphabet + '_' + character
					image_original_labels.append(original_label)
					image_path = os.path.join(character_path, image)
					dest_path = os.path.join(sub_organized_dir, str(current_id) + '.png')
					image_paths.append(dest_path)
					if os.path.isfile(dest_path) == False:
						copyfile(image_path, dest_path)
					img = transform(np.asarray(Image.open(image_path)).astype('uint8'))
					if images_torch == None:
						images_torch = img
					else:
						images_torch = torch.cat([images_torch, img])
					if current_id %200 == 0:
						print('Progress {}/19800'.format(current_id))
					current_id +=1
				current_label +=1
		result_dict = {'Image original name' : image_names,
					   'Image current id' : image_ids,
					   'Image original_label' : image_original_labels, 
					   'Image current label' : labels,
					   'Image path' : image_paths}
		result_dset = pd.DataFrame(result_dict)
		if data_type == 'images_background':
			csv_path = os.path.join(organized_dir, 'train_description.csv')
			labels_torch = torch.tensor(labels)
			train_set = [images_torch, labels_torch]
			torch.save(train_set, os.path.join(organized_dir, 'train_set.pt'))
		else:
			csv_path = os.path.join(organized_dir, 'test_description.csv')
			labels_torch = torch.tensor(labels)
			test_set = [images_torch, labels_torch]
			torch.save(test_set, os.path.join(organized_dir, 'test_set.pt'))
		result_dset.to_csv(csv_path, index = False)
		
		with open('./data/omniglot-py/organized_data/pt_sets/.gitignore', 'w') as f:
			print('train_set.pt', file = f)
			print('test_set.pt', file = f)
		f.close
		with open('./data/omniglot-py/organized_data/.gitignore', 'w') as f:
			print('pt_sets', file = f)
		f.close
		with ZipFile('./data/omniglot-py/organized_data/pt_sets.zip', 'w') as zipObj:
			zipObj.write('./data/omniglot-py/organized_data/pt_sets/train_set.pt')
			zipObj.write('./data/omniglot-py/organized_data/pt_sets/test_set.pt')
		zipObj.close()

		print('Omniglot downloaded and organized')

def create_pt_file(dataloader, save_path = './bla.pt'):
	#for i, (X,Y) in enumerate(dataloader):
	for i, (X, Y) in enumerate(dataloader):
		if i%(len(dataloader)/10)==0:
			print('Preprocessing progress {}%'.format(i/(len(dataloader)/100)))
		if i==0:
			data = X
			labels = Y
		elif i>0:
			data = torch.cat([data, X])
			labels = torch.cat([labels, Y])
	return data, labels

def unpickle(file):
	with open(file, 'rb') as fo:
		res = pickle.load(fo, encoding='bytes')
	return res
