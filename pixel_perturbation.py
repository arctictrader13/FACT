
import torch
from torchvision import datasets, transforms, utils
import os

# Import saliency methods and models
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from models.vgg import *
from models.resnet import *
from misc_functions import *
import argparse
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import random

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
data_PATH= PATH + 'dataset/'

batch_size = 1
device = "cpu"



def main():
	# same transformations for each dataset
	transform_standard = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
							 std=[0.229, 0.224, 0.225]), ])

	if ARGS.dataset == "cifar":
		dataset = data_PATH + "/cifar/"
		data = torchvision.datasets.CIFAR100(dataset, train=True, transform=transform_standard,target_transform=None,
											 download=True)
		# Dataset loader for sample images
		sample_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
	elif ARGS.dataset == "mnist":
		dataset = data_PATH + "/mnist/"
		data = torchvision.datasets.MNIST(dataset, train=True, transform=transform_standard,target_transform=None,
										  download=True)
		# Dataset loader for sample images
		sample_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

	elif ARGS.dataset == "imagenet":
		dataset = data_PATH
		sample_loader = torch.utils.data.DataLoader(
			datasets.ImageFolder(dataset, transform=transforms.Compose([
				transforms.Resize((224, 224)),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
			])),
			batch_size=batch_size, shuffle=False)


	unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
							   		std = [0.229, 0.224, 0.225])
	
	if ARGS.model == "resnet":
		model_class = ARGS.model + ARGS.model_type
		model = resnet18(pretrained=True)

	#model = vgg16_bn(pretrained=True)
	model = resnet18(pretrained=True)

	# Initialize FullGrad objects
	fullgrad = FullGrad(model)
	# simple_fullgrad = SimpleFullGrad(model)

	save_path = PATH + 'results/' + ARGS.dataset

	results_dict = {}

	total_features = 224*224

	for grad_type in ARGS.grads:
		means = []
		stds = []

		for i in ARGS.k:
			k_most_salient = int(i*total_features)
			counter = 0
			tmp_results = []

			for batch_idx, (data, target) in enumerate(sample_loader):
				# for debugging purposes
				counter += 1
				if counter % 50 == 0:
					print("{} images processed".format(counter))
				if counter == 200:
					break

				data, target = data.to(device).requires_grad_(), target.to(device)

				# Run Input through network
				initial_output = model.forward(data)

				# change pixels based on grad method
				if ARGS.grads == "random":
					sample_seeds = np.random.random_integers(0, 10000, 10)
					for seed in sample_seeds:
						random.seed(seed)
						tmp_results = abs_frac_per_grad(model, data, target, grad_type, k_most_salient, initial_output, tmp_results)
						print(tmp_results)

				else:
					tmp_results = abs_frac_per_grad(model, data, target, grad_type, k_most_salient, initial_output, tmp_results, fullgrad)

			#print("Absolute fractional output change: ", tmp_result)
			#print("Actual values: ",  initial_class_probability, final_class_probability)

			means.append(np.mean(tmp_results))
			stds.append(np.std(tmp_results))
		results_dict[grad_type] = [means, stds]
	print(results_dict)
	plot_all_grads(results_dict)

def abs_frac_per_grad(model, data, target, grad_type, k, initial_output, tmp_results, grad=None):

	data = change_pix_per_grad(data, target, grad_type, k, grad)

	final_output = model.forward(data)
	initial_class_probability, predicted_class = initial_output.max(1)
	final_class_probability = final_output[0, predicted_class]

	tmp_result = abs(final_class_probability - initial_class_probability) / initial_class_probability
	tmp_results.append(round(tmp_result.item(), 5))

	return tmp_results

def plot_all_grads(results_dict):
	plt.figure()
	axes = plt.gca()
	axes.set_xlim([0, 0.1])
	axes.set_ylim([0, 1])
	axes.set_xlabel('% pixels removed')
	axes.set_ylabel('Absolute fractional output change')

	for key, v in results_dict.items():

		# Plot the mean and variance of the predictive distribution on the 100000 data points.
		plt.plot(ARGS.k, np.array(v[0]), linewidth = 0.5, label = str(key))
		plt.fill_between(ARGS.k, np.array(v[0]) - np.array(v[1]), np.array(v[0]) + np.array(v[1]), alpha = 1/2)
	plt.legend()
	plt.show()

def change_pix_per_grad(data, target, grad_type, k, grad):
	[column_size, row_size] = data.size()[2:4]

	# Compute saliency maps for the input data
	if grad_type=="fullgrad":
		cam = grad.saliency(data)

	elif grad_type == "random":
		for index in random.sample(range(column_size*row_size), k):
			row = int(index / row_size)
			column = index % row_size
			data[0, 0, row, column] = 1.0
		return data

	elif grad_type == "gradcam":
		gcam = GradCAM(model=model)
		_ = gcam.forward(images)
		# Grad-CAM
		gcam.backward(ids=ids[:, [i]])
		regions = gcam.generate(target_layer=target_layer)

	for index in torch.topk(cam.view((-1)), k=k, largest=True)[1]:
		row = index / row_size
		column = index % row_size
		data[:, :, row, column] = 1.0
	return data




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--k', default= [0.01, 0.05, 0.1, 0.3], type=float,
						help='Percentage of k% most salient pixels')
	parser.add_argument('--most_salient', default=True, type=bool,
						help='most salient = True or False depending on retrain or pixel perturbation')
	parser.add_argument('--model', default="resnet", type=str,
						help='which model to use')
	parser.add_argument('--model_type', default="18", type=str,
						help='which model type: resnet_18, ...')
	parser.add_argument('--dataset', default="imagenet", type=str,
						help='which dataset')
	parser.add_argument('--grads', default=["random", "fullgrad"], type=str,
						help='which grad methods to be applied')
	ARGS = parser.parse_args()
	main()