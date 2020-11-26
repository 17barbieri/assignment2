import os
import pdb
import numpy as np
import torchvision
from torchvision.datasets import Omniglot
from os.path import join
from shutil import copyfile
import pandas as pd
import torch
from random import sample
from PIL import Image
from zipfile import ZipFile

def download_omniglot():
	if os.path.isdir('../data/omniglot-py/images_background') == False or os.path.isdir('./data/omniglot-py/images_evaluation') == False:
		train_dataset = Omniglot(root="../data",
								 download=True,
								 background = True,
								 transform=torchvision.transforms.ToTensor()
								 )
		test_dataset = Omniglot(root="../data",
								download=True,
								background = False,
								transform=torchvision.transforms.ToTensor()
								)
	#Organizing data
	data_dir = '../data/omniglot-py'
	transform = torchvision.transforms.Compose([
								torchvision.transforms.ToTensor(),
								torchvision.transforms.Normalize(
								(0,), (1,))
								])
	for data_type in ['images_background', 'images_evaluation']:
		organized_dir = '../data/omniglot-py/organized_data'
		if data_type == 'images_background':
			sub_organized_dir = join(organized_dir, 'train')
		else:
			sub_organized_dir = join(organized_dir, 'test')
		zip_name = join(organized_dir, 'pt_sets.zip')
		if os.path.isfile(join(organized_dir, 'pt_sets/train_set.pt')) == True and os.path.isfile(join(organized_dir, 'pt_sets/test_set.pt')) == True :
			print('PT files already existing')
			break
		if os.path.isfile(zip_name):
			print('Extracting zip archive')
			with ZipFile(zip_name, 'r') as zip:
				zip.extractall(organized_dir)
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
		alphabets = sorted(os.listdir(join(data_dir, data_type)))
		for alphabet in alphabets:
			alphabet_path = join(data_dir, data_type, alphabet)
			characters = sorted(os.listdir(alphabet_path))
			for character in characters:	
				character_path = join(alphabet_path, character)
				images = sorted(os.listdir(character_path))
				for image in images:
					image_names.append(image)
					image_ids.append(current_id)
					labels.append(current_label)
					original_label = alphabet + '_' + character
					image_original_labels.append(original_label)
					image_path = join(character_path, image)
					dest_path = join(sub_organized_dir, str(current_id) + '.png')
					image_paths.append(dest_path)
					if os.path.isfile(dest_path) == False:
						copyfile(image_path, dest_path)
					img = transform(np.asarray(Image.open(image_path)).astype('uint8'))
					if images_torch == None:
						images_torch = img
					else:
						images_torch = torch.cat([images_torch, img])
					if current_id %200 == 0:
						print(current_id)
					current_id +=1
				current_label +=1
		result_dict = {'Image original name' : image_names,
					   'Image current id' : image_ids,
					   'Image original_label' : image_original_labels, 
					   'Image current label' : labels,
					   'Image path' : image_paths}
		result_dset = pd.DataFrame(result_dict)
		if data_type == 'images_background':
			csv_path = join(organized_dir, 'train_description.csv')
			labels_torch = torch.tensor(labels)
			train_set = [images_torch, labels_torch]
			torch.save(train_set, join(organized_dir, 'train_set.pt'))
		else:
			csv_path = join(organized_dir, 'test_description.csv')
			labels_torch = torch.tensor(labels)
			test_set = [images_torch, labels_torch]
			torch.save(test_set, join(organized_dir, 'test_set.pt'))
		result_dset.to_csv(csv_path, index = False)
	print('Omniglot downloaded and organized')

def sample_class_labels(classes_to_load, labels_initial, data_percentage):
	cls_indexes = []
	for j in classes_to_load:
		class_indexes = (labels_initial==j).nonzero()
		class_indexes = sample(class_indexes.view(-1).tolist(), int(len(class_indexes)*data_percentage))
		cls_indexes.append(class_indexes)
	return cls_indexes

class omniglot_dataset:
	def __init__(self, classes_to_load, data_percentage, train = True):
		download_omniglot()
		self.classes_to_load = classes_to_load
		self.data_percentage = data_percentage
		if train == True:
			self.data_initial, self.labels_initial = torch.load('../data/omniglot-py/organized_data/pt_sets/train_set.pt')
		if train == False:
			self.data_initial, self.labels_initial = torch.load('../data/omniglot-py/organized_data/pt_sets/test_set.pt')

		self.cls_indexes=sample_class_labels(self.classes_to_load, self.labels_initial, self.data_percentage)

		self.data = self.data_initial[np.concatenate(self.cls_indexes)].float()
		self.labels = self.labels_initial[np.concatenate(self.cls_indexes)].long()

	def __len__(self):
		return len(self.data), len(set(self.labels.numpy().tolist()))

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]

if __name__ == '__main__':
	dataset = omniglot_dataset(np.arange(0,13800), 1)
	pdb.set_trace()