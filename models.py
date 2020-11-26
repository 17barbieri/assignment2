import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class first_model_MNIST(nn.Module):
	# define model elements
	def __init__(self, n_classes):
		super(first_model_MNIST, self).__init__()
		self.layer1 = [nn.Linear(28*28, 64), nn.ReLU()]
		self.layer2 = [nn.Linear(64, 32), nn.ReLU()]
		self.layer1 = nn.Sequential(*self.layer1)
		self.layer2 = nn.Sequential(*self.layer2)
		self.classifier = nn.Linear(32, n_classes)
 
	# forward propagate input
	def forward(self, X):
		X = X.view(X.size(0),-1)
		X_1 = self.layer1(X)
		features = self.layer2(X_1)
		logits = self.classifier(features)
		
		Y_hat = torch.topk(logits, 1, dim = 1)[1]
		prob = F.softmax(logits, dim=0)
		return features, logits, prob, Y_hat # X_2 is here the embbeding layer

class first_model_CIFAR(nn.Module):
	# define model elements
	def __init__(self, n_classes):
		super(first_model_CIFAR, self).__init__()
		self.layer1 = [nn.Linear(32*32*3, 64), nn.ReLU()]
		self.layer2 = [nn.Linear(64, 32), nn.ReLU()]
		self.layer1 = nn.Sequential(*self.layer1)
		self.layer2 = nn.Sequential(*self.layer2)
		self.classifier = nn.Linear(32, n_classes)
 
	# forward propagate input
	def forward(self, X):
		X = X.view(X.size(0),-1)
		X_1 = self.layer1(X)
		features = self.layer2(X_1)
		logits = self.classifier(features)
		
		Y_hat = torch.topk(logits, 1, dim = 1)[1]
		prob = F.softmax(logits, dim=0)
		return features, logits, prob, Y_hat # X_2 is here the embbeding layer

class first_model_Omniglot(nn.Module):
	# define model elements
	def __init__(self, n_classes):
		super(first_model_Omniglot, self).__init__()
		self.layer1 = [nn.Linear(32*32*3, 64), nn.ReLU()]
		self.layer2 = [nn.Linear(64, 32), nn.ReLU()]
		self.layer1 = nn.Sequential(*self.layer1)
		self.layer2 = nn.Sequential(*self.layer2)
		self.classifier = nn.Linear(32, n_classes)
 
	# forward propagate input
	def forward(self, X):
		X = X.view(X.size(0),-1)
		X_1 = self.layer1(X)
		features = self.layer2(X_1)
		logits = self.classifier(features)
		
		Y_hat = torch.topk(logits, 1, dim = 1)[1]
		prob = F.softmax(logits, dim=0)
		return features, logits, prob, Y_hat # X_2 is here the embbeding layer

class LeNet5_Omniglot(nn.Module):
	def __init__(self, n_classes, img_channels = 1):
		super(LeNet5_Omniglot, self).__init__()
		self.feature_extractor = nn.Sequential(            
			nn.Conv2d(in_channels=img_channels, out_channels=6, kernel_size=7, stride=1),
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=2),
			nn.Conv2d(in_channels=6, out_channels=16, kernel_size=7, stride=1),
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=2),
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=2),
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
			nn.Tanh()
		)

		self.classifier = nn.Sequential(
			nn.Linear(in_features=1024, out_features=84),
			nn.Tanh(),
			nn.Linear(in_features=84, out_features=n_classes),
		)


	def forward(self, x):
		features = self.feature_extractor(x)
		x = torch.flatten(features, 1)
		logits = self.classifier(x)
		Y_hat = torch.topk(logits, 1, dim = 1)[1]
		probs = F.softmax(logits, dim=1)
		return features.view(features.size(0), -1), logits, probs, Y_hat

class LeNet5_MNIST(nn.Module):
	def __init__(self, n_classes, img_channels = 1):
		super(LeNet5_MNIST, self).__init__()
		self.feature_extractor = nn.Sequential(            
			nn.Conv2d(in_channels=img_channels, out_channels=6, kernel_size=5, stride=1),
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=2),
			nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=2),
			nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1),
			nn.Tanh()
		)

		self.classifier = nn.Sequential(
			nn.Linear(in_features=120, out_features=84),
			nn.Tanh(),
			nn.Linear(in_features=84, out_features=n_classes),
		)


	def forward(self, x):
		features = self.feature_extractor(x)
		x = torch.flatten(features, 1)
		logits = self.classifier(x)
		Y_hat = torch.topk(logits, 1, dim = 1)[1]
		probs = F.softmax(logits, dim=1)
		return features.view(features.size(0), -1), logits, probs, Y_hat

class LeNet5_CIFAR(nn.Module):
	def __init__(self, n_classes, img_channels = 1):
		super(LeNet5_CIFAR, self).__init__()
		self.feature_extractor = nn.Sequential(            
			nn.Conv2d(in_channels=img_channels, out_channels=6, kernel_size=5, stride=1),
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=2),
			nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=2),
			nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
			nn.Tanh()
		)

		self.classifier = nn.Sequential(
			nn.Linear(in_features=120, out_features=84),
			nn.Tanh(),
			nn.Linear(in_features=84, out_features=n_classes),
		)


	def forward(self, x):
		features = self.feature_extractor(x)
		x = torch.flatten(features, 1)
		logits = self.classifier(x)
		Y_hat = torch.topk(logits, 1, dim = 1)[1]
		probs = F.softmax(logits, dim=1)
		return features.view(features.size(0), -1), logits, probs, Y_hat