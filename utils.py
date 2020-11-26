import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from random import sample
import pdb

def collect_features(model, dataloader):
	model.eval()
	for i, (X, Y) in enumerate(dataloader):
		features, logits, prob, Y_hat = model(X)
		if i == 0:
			all_features = features
			all_labels = Y
		elif i > 0:
			all_features = torch.cat([all_features, features], dim = 0)
			all_labels = torch.cat([all_labels, Y], dim = 0)
	return all_features, all_labels

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0, base_loss_fn = nn.CrossEntropyLoss()):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.base_loss_fn = base_loss_fn

    def forward(self, features1, logits1, label1, features2, logits2, label2):
        loss_1 = self.base_loss_fn(logits1, label1)
        loss_2 = self.base_loss_fn(logits2, label2)
        label = label1.eq(label2).type(torch.uint8)
        euclidean_distance = F.pairwise_distance(features1, features2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_1 + loss_2 + loss_contrastive

def create_figure(epochs, history, save_path, title = 'Training loss monitoring', fig = False):
	plt.plot(np.arange(epochs), history)
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title(title)
	plt.savefig(os.path.join(save_path, title + '.png'))
	if fig:
		plt.show()
	plt.clf()

def plot_PCA(all_features, all_labels, save_path, title = 'PCA_plot', fig = False, nb_to_plot = 15):
	#Distinct color generator : https://mokole.com/palette.html
	colors = ['darkgreen', 'darkblue', 'darkslategray', 'orangered', 'yellow', 'lime', 'aqua', 'fuchsia', 'hotpink', 'moccasin']
	for i, class_nb in enumerate(set(all_labels)):
		class_nb = int(class_nb)
		cls_idx = (all_labels==class_nb).nonzero()
		cls_idx = sample(cls_idx[0].tolist(), nb_to_plot)
		if i == 0:
			features_to_plot = all_features[cls_idx]
			labels_to_plot = all_labels[cls_idx]
		else:
			features_to_plot = np.concatenate((features_to_plot, all_features[cls_idx]))
			labels_to_plot = np.concatenate((labels_to_plot, all_labels[cls_idx]))

	pca = PCA(n_components=2)
	projection_2D = pca.fit(features_to_plot).transform(features_to_plot)
	for i, (point, label) in enumerate(zip(projection_2D, labels_to_plot)):
		x_coord, y_coord = point
		plt.scatter(x_coord, y_coord, alpha=.8, c=colors[label], label = label)
	plt.savefig(os.path.join(save_path, title + '.png') )
	if fig:
		plt.show()
	plt.clf()