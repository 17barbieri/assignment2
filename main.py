import os
import pdb
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from random import sample

from models import first_model_MNIST, first_model_CIFAR, first_model_Omniglot, LeNet5_MNIST, LeNet5_CIFAR, LeNet5_Omniglot
from datasets.load_datasets import load_datasets
from train_utils import train, evaluate, one_shot_testing
from utils import collect_features, create_figure, plot_PCA, ContrastiveLoss

parser = argparse.ArgumentParser(description='Configurations one-shot-learning testing')
parser.add_argument('--model', type =str, default='LeNet5', help= 'Select the model you want to use')
parser.add_argument('--epochs', type= int, default=40, help= 'Number of training epochs')
parser.add_argument('--fig', action = 'store_true', default = False, help='If true, plots the figures, saves it in the experiment folder')
parser.add_argument('--few_shot', action = 'store_true', default = False, help = 'If true, the test is done with few-shot learning (see argument nb_few_shot)')
parser.add_argument('--nb_plot_points', type=int, default=15, help= 'The number of points per class we plot in PCA')
parser.add_argument('--nb_few_shot', type=int, default=15, help= 'The number of labeled points we provide to find the label of the cluster for few-shot-learning')
parser.add_argument('--dataset', type = str, default= 'CIFAR100')
parser.add_argument('--data_percent', type=float, default=.6)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--n_classes', type=int, default=5, help= 'The number of classes on which we train')
parser.add_argument('--loss', type = str, default = 'contrastive')
parser.add_argument('--lr', type = float, default = 0.0001)
parser.add_argument('--margin', type = float, default = 2.0)
parser.add_argument('--experiment_title', type = str, default = None)
parser.add_argument('--experimentation', action = 'store_true', default = False)
parser.add_argument('--save_path', type = str, default = None, help = 'Path where to save the experiment')
args = parser.parse_args()

def main(args):
	#Creating save directory :
	if args.save_path is None:
		save_path = os.path.join(os.curdir, 'results', args.dataset + '_dataset', args.loss + '_loss', 'margin_' + str(args.margin) + '_' +  str(args.model) + '_nb_classes_' + str(args.n_classes) + '_data_percent_' + str(int(args.data_percent*100)))
	else:
		save_path = args.save_path
	os.makedirs(save_path, exist_ok = True)
		
	#Creating settings file
	settings = {'One/few-shot': args.few_shot,
				'Dataset': args.dataset,
		 		'Model' : args.model,
		 		'Number of classes we trianed on' : args.n_classes,
		 		'Epochs number': args.epochs,
		 		'Batch size' : args.batch_size,
		 		'Loss' : args.loss,
		 		'Percentage of data used for training': args.data_percent,
		 		'Number points used for initial one-shot learning' : args.nb_few_shot,
		 		'learning rate': args.lr,
		 		'save_path': args.save_path
		 		}

	if args.loss == 'contrastive':
		settings.update({'margin' : args.margin})
	if args.experiment_title is not None:
		settings.update({'Experiment title' : args.experiment_title})

	if args.experiment_title is None:
		settings_path = save_path + '/Experiment_arguments.txt'
	else:
		settings_path = save_path + '/{}.txt'.format(args.experiment_title)
	with open(settings_path, 'w') as f:
		print(settings, file=f)
	f.close()

	train_bool = True
	if 'model_ckpt.pt' in os.listdir(save_path):
		train_bool = False
		print('Model already trained')
	if train_bool:
		print('Model has not yet been trained for the requested parameters.')
		print('Running {} dataset, on {}% percent of the available data'.format(args.dataset, int(args.data_percent*100)))
		print("Training on {} epochs: ".format(args.epochs))
	print('\n')

	#Select dataset
	if train_bool:
		train_loader, test_loader = load_datasets(args.dataset, args.batch_size, args.data_percent, args.n_classes, train_classes = True)
	one_shot_train_loader, one_shot_test_loader = load_datasets(args.dataset, args.batch_size, args.data_percent)
	
	if args.dataset == 'MNIST': 
		input_channels = 1
		if args.model == 'first_model':
			model = first_model_MNIST(args.n_classes)
			if not train_bool:
				print('Loading trained model')
				model.load_state_dict(torch.load(os.path.join(save_path, 'model_ckpt.pt')))
		elif args.model == 'LeNet5':
			model = LeNet5_MNIST(args.n_classes, input_channels)
			if not train_bool:
				print('Loading trained model')
				model.load_state_dict(torch.load(os.path.join(save_path, 'model_ckpt.pt')))
		else:
			raise NotImplementedError	
	elif args.dataset == 'CIFAR100':
		input_channels = 3
		if args.model == 'first_model':
			model = first_model_CIFAR(args.n_classes)
			if not train_bool:
				print('Loading trained model')
				model.load_state_dict(torch.load(os.path.join(save_path, 'model_ckpt.pt')))
		elif args.model == 'LeNet5':
			model = LeNet5_CIFAR(args.n_classes, input_channels)
			if not train_bool:
				print('Loading trained model')
				model.load_state_dict(torch.load(os.path.join(save_path, 'model_ckpt.pt')))
		else:
			raise NotImplementedError	
	elif args.dataset == 'Omniglot':
		input_channels = 1
		if args.model == 'first_model':
			model = first_model_Omniglot(args.n_classes)
			if not train_bool:
				print('Loading trained model')
				model.load_state_dict(torch.load(os.path.join(save_path, 'model_ckpt.pt')))
		elif args.model == 'LeNet5':
			model = LeNet5_Omniglot(args.n_classes, input_channels)
			if not train_bool:
				print('Loading trained model')
				model.load_state_dict(torch.load(os.path.join(save_path, 'model_ckpt.pt')))
		else:
			raise NotImplementedError	
		
	#Define Loss
	if args.loss=='CE':
		loss_fn = nn.CrossEntropyLoss()
	elif args.loss == 'contrastive':
		loss_fn = ContrastiveLoss(margin = args.margin)
	else:
		raise NotImplementedError
	
	#Define optimizer
	optimizer = optim.Adam(model.parameters(), lr = args.lr)
		
	#Train model
	if train_bool:
		loss_history, acc_history, trained_model = train(model, train_loader, optimizer, loss_fn, args.epochs)
		#Save model to re-use
		torch.save(trained_model.state_dict(), os.path.join(save_path, 'model_ckpt.pt'))

		#Create training figures
		create_figure(args.epochs, loss_history, save_path, 'Train loss monitoring', args.fig)
		create_figure(args.epochs, acc_history, save_path, 'Train acc monitoring', args.fig)

		#Run on test set
		test_loss, test_acc = evaluate(trained_model, test_loader, loss_fn)
		print('Test loss {:.3f},\nTest accuracy {:.3f}'.format(test_loss, test_acc))

	#Running on all the dataset and clustering
	#Step 1: get all embedded representations
	all_features, all_labels = collect_features(model, one_shot_test_loader)
	all_features, all_labels = all_features.detach().numpy(), all_labels.detach().numpy()
	
	#Step 2: Computing PCA and plotting, only for MNIST as it has a reasonable number of classes
	if args.dataset == 'MNIST':
		plot_PCA(all_features, all_labels, save_path, fig = args.fig, nb_to_plot = args.nb_plot_points)
	
	# Running one/few-shot test
	print('Running one/few-shot prediction')
	if args.few_shot == False:
		args.nb_few_shot = 1
	total_accuracy, per_class_accuracy = one_shot_testing(model, one_shot_test_loader, args.nb_few_shot)
	print('Overall accuracy is: {:.3f}'.format(total_accuracy))
	unknown_classes_total_accuracy = np.mean(per_class_accuracy[args.n_classes:])
	print('Unknown classes overall accuracy is: {:.3f}'.format(unknown_classes_total_accuracy))
	
	for i in range(len(per_class_accuracy)):
		print('Accuracy for class {} is: {}'.format(i, per_class_accuracy[i]))
	results = {'Overall accuracy': [total_accuracy], 'Unknown classes overall_acc' : [unknown_classes_total_accuracy], 'Per class accuracy' : [str(per_class_accuracy)] }
	dset = pd.DataFrame(results)
	dset.to_csv(os.path.join(save_path, 'results.csv'), index = False)
	return total_accuracy, unknown_classes_total_accuracy, per_class_accuracy

if __name__ == '__main__':

	if args.experimentation:
		dset = None
		if args.experiment_title is None:
			print('Error: missing Experiment title')
			raise NotImplementedError
		experiment_save_path = './' + args.experiment_title + '.csv'
		
		for param in np.linspace(0.1, 1, 10):
			args.data_percent = param
			total_accuracy, unknown_classes_total_accuracy, per_class_accuracy = main(args)
			
			# Creating experiment summary 
			if dset is None:
				initial_dict = {'param value' : [param], 'Overall accuracy': [total_accuracy], 'Unknown classes overall_acc' : [unknown_classes_total_accuracy], 'Per class accuracy' : [str(per_class_accuracy)] }
				dset = pd.DataFrame(initial_dict)
				dset.to_csv(experiment_save_path, index = False)
			else:
				dset = pd.read_csv(experiment_save_path)
				new_row = pd.DataFrame({'param value' : [param], 'Overall accuracy': [total_accuracy], 'Unknown classes overall_acc' : [unknown_classes_total_accuracy], 'Per class accuracy' : [str(per_class_accuracy)] })
				dset = pd.concat([dset, new_row], ignore_index = True)
				dset.to_csv(experiment_save_path, index = False)

	else:
		main(args)