import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import blackbox_synthetic_GIN



def eval(model):

	pickle_in = open("save/graphs/tree_cycle_star_eval.pkl","rb")
	data = pickle.load(pickle_in)
	pickle_in.close()
	
	adj_list = data[0]
	edge_indicies_list = data[1]
	features_list = data[2] 
	labels_list = data[3]
	class_distribution = data[4]

	num_classes = len(class_distribution)
	num_correct = 0
	total = 0

	accuracy_per_class = [0 for i in range(num_classes)] #blackbox accuracy per class
	num_labels_per_class = [0 for i in range(num_classes)]

	for i in range(len(edge_indicies_list)):

		logits = model(features_list[i], edge_indicies_list[i])
		_, labels = torch.max(logits, 1)

		#Get number of correct labels per class
		for c in range(num_classes):

			ground_truth_c_indicies = (labels_list[i] == c).nonzero()
			matching_c = (labels[ground_truth_c_indicies] == c).sum().item()
			accuracy_per_class[c] += matching_c
			num_labels_per_class[c] += ground_truth_c_indicies.shape[0]


		num_correct += (labels == labels_list[i].long()).sum().item()
		total += labels.shape[0]

	accuracy = num_correct/total

	for c in range(num_classes):

		 accuracy_per_class[c] /= num_labels_per_class[c]

	return accuracy, accuracy_per_class


def trainBlackbox():

	print("Reading Data...")

	pickle_in = open("save/graphs/tree_cycle_star_train.pkl","rb")
	data = pickle.load(pickle_in)
	pickle_in.close()
	
	adj_list = data[0]
	edge_indicies_list = data[1]
	features_list = data[2] 
	labels_list = data[3]
	class_distribution = data[4]

	num_features = features_list[0].shape[1]
	num_classes = len(class_distribution)

	model = blackbox_synthetic_GIN(num_features, num_classes)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
	model.train()

	epochs = 200

	print("Training Model...")

	for e in range(epochs):

		for i in range(len(edge_indicies_list)):

			optimizer.zero_grad()

			logits = model(features_list[i], edge_indicies_list[i])
			logits = F.log_softmax(logits, dim=-1)

			loss = F.nll_loss(logits, labels_list[i].long())

			loss.backward()
			optimizer.step()

		if(e%50 == 0):
			print("Loss at epoch", e, ":", loss)

	#Calculate Accuracy on eval dataset
	model.eval()
	accuracy, accuracy_per_class = eval(model)

	print("Accuracy:", accuracy)
	print("Accuracy per Class:", accuracy_per_class)


	#Save blackbox model
	save_path = "save/tree_cycle_star_model.pkl"
	torch.save({"model":model.state_dict()}, save_path)


trainBlackbox()