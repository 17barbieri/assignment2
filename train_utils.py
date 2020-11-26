import pdb
import torch
import numpy as np

from random import sample
from sklearn.metrics import accuracy_score as accuracy

from utils import ContrastiveLoss, collect_features

def train(model, dataloader, optimizer, loss_fn, epochs):
	model.train()
	loss_history = []
	acc_history = []
	for epoch in range(epochs):
		print('Running epoch: ' + str(epoch))
		epoch_train_loss = 0.
		epoch_train_acc = 0.
		
		for i, (X, Y) in enumerate(dataloader):
			optimizer.zero_grad()
			
			features, logits, prob, Y_hat = model(X)
			
			if isinstance(loss_fn, ContrastiveLoss):
				# Creating shuffled pairs
				shuffled_idx = np.arange(len(features))
				np.random.shuffle(shuffled_idx)

				features_2 = features[shuffled_idx]
				logits_2 = logits[shuffled_idx]
				Y_2 = Y[shuffled_idx]
				
				#Computing the contrastive loss
				loss = loss_fn(features, logits, Y, features_2, logits_2, Y_2)

			else: # If normal loss (CE, MSE...)
				loss = loss_fn(logits, Y)
			
			acc = accuracy(Y_hat, Y)
			epoch_train_loss += loss.item()
			epoch_train_acc += acc
			loss.backward()
			optimizer.step()
		
		#Averaging loss and accuracy over the epoch
		epoch_train_loss/=len(dataloader)
		epoch_train_acc/=len(dataloader)
		
		print('Epoch_loss: {:.3f}'.format(epoch_train_loss))
		print('Epoch_acc: {:.3f}'.format(epoch_train_acc))
		print('\n')
		
		# Storing the history
		loss_history.append(epoch_train_loss)
		acc_history.append(epoch_train_acc)
	return loss_history, acc_history, model

def evaluate(model, dataloader, loss_fn):
	model.eval()
	test_acc, test_loss = 0., 0.
	for i, (X, Y) in enumerate(dataloader):
		features, logits, prob, Y_hat = model(X)
		if isinstance(loss_fn, ContrastiveLoss):
			shuffled_idx = np.arange(len(features))
			np.random.shuffle(shuffled_idx)
			features_2 = features[shuffled_idx]
			logits_2 = logits[shuffled_idx]
			Y_2 = Y[shuffled_idx]
			loss = loss_fn(features, logits, Y, features_2, logits_2, Y_2)
		else:
			loss = loss_fn(logits, Y)
		acc = accuracy(Y_hat, Y)
		test_loss += loss.item()
		test_acc += acc
	test_loss/=len(dataloader)
	test_acc/=len(dataloader)

	return test_loss, test_acc

def one_shot_testing(model, dataloader, nb_initial_examples):
	# Extract features representations of all samples
	print('Collecting features...', end = '')
	all_features, all_labels = collect_features(model, dataloader)
	print('Done')

	unique_label_list = set(all_labels.detach().numpy())
	correct_prediction_per_class = [0.]*len(unique_label_list)
	nb_element_per_class = [0.]*len(unique_label_list)
	
	# Selecting the points with known label for one/few-shot learning
	representants = torch.empty(len(unique_label_list), nb_initial_examples, all_features.size(1))
	for label in unique_label_list:
		#For each class of the dataset, select "nb_initial_examples" labeled representants
		cls_idx = sample((all_labels==label).nonzero().view(-1).tolist(), nb_initial_examples)
		representants[label] = all_features[cls_idx]

	# Predicting a label for each point
	for i, feature in enumerate(all_features):
		if i%(int(len(all_features)/10)) ==0:
			print('Prediction progress {}%'.format(int(i/(len(all_features)/100))))
		dists_to_class_representants = []
		
		# Calculate distance to each cluster
		for label in unique_label_list:
			mean_representant = torch.mean(representants[label], dim = 0)
			dists_to_class_representants.append(torch.norm(feature-mean_representant, 2).detach().numpy().tolist())
		
		# Predict the label by taking the minimal distance to the clusters
		Y_hat = dists_to_class_representants.index(min(dists_to_class_representants))
		
		# Update performance
		Y = all_labels[i]
		correct_prediction_per_class[Y]+= int(Y_hat == Y)
		nb_element_per_class[Y]+=1
	
	# Compute accuracy
	per_class_accuracy = [i/j for (i, j) in zip(correct_prediction_per_class,nb_element_per_class)]
	total_accuracy = sum(correct_prediction_per_class)/len(all_labels)

	return total_accuracy, per_class_accuracy