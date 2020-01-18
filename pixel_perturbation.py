
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
# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
data_PATH= PATH + 'dataset/'

batch_size = 1
# percentages of most salient pixels to be removed
k = [0.01, 0.05]

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
	elif ARGS.dataset == "mnist":
		dataset = data_PATH + "/mnist/"
		data = torchvision.datasets.MNIST(dataset, train=True, transform=transform_standard,target_transform=None,
										  download=True)


	# Dataset loader for sample images
	sample_loader = torch.utils.data.DataLoader(data, batch_size= batch_size, shuffle=False)
	
	unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
							   		std = [0.229, 0.224, 0.225])
	
	if ARGS.model == "resnet":
		model_class = ARGS.model + ARGS.model_type
		model = resnet18(pretrained=True)

	#model = vgg16_bn(pretrained=True)
	model = resnet18(pretrained=True)

	# Initialize FullGrad objects
	fullgrad = FullGrad(model)
	simple_fullgrad = SimpleFullGrad(model)

	save_path = PATH + 'results/' + "/" + ARGS.dataset

	results_dict = {}

	total_featues = 224*224

	for i in k:
		k_most_salient = int(i*total_featues)
		counter = 0
		results_dict[str(i)] = []

		for batch_idx, (data, target) in enumerate(sample_loader):
			# for debugging purposes
			counter += 1
			if counter == 3:
				break

			data, target = data.to(device).requires_grad_(), target.to(device)

			# Run Input through network
			initial_output = model.forward(data)

			# Compute saliency maps for the input data
			cam = fullgrad.saliency(data)
			cam_simple = simple_fullgrad.saliency(data)

			[column_size, row_size] = data.size()[2:4]
			for index in torch.topk(cam.view((-1)), k=k_most_salient, largest=True)[1]:
				row = index / row_size
				column = index % row_size
				data[0, 0, row, column] = 1.0

			final_output = model.forward(data)
			initial_class_probability, predicted_class = initial_output.max(1)
			final_class_probability = final_output[0, predicted_class]
			tmp_result = abs(final_class_probability - initial_class_probability) / initial_class_probability
			results_dict[str(i)].append(round(tmp_result.item(), 4))
			print("Absolute fractional output change: ", tmp_result)
			#print("Actual values: ",  initial_class_probability, final_class_probability)
	print(results_dict)
	# TODO ADD PLOT FUNCTION: x axis k values, y axis absolute fractional output change, values: mean with std as shading

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--k', default=1, type=int,
						help='Percentage of k% most salient pixels')
	parser.add_argument('--most_salient', default=True, type=bool,
						help='most salient = True or False depending on retrain or pixel perturbation')
	parser.add_argument('--model', default="resnet", type=str,
						help='which model to use')
	parser.add_argument('--model_type', default="18", type=str,
						help='which model type: resnet_18, ...')
	parser.add_argument('--dataset', default="cifar", type=str,
						help='which dataset')
	ARGS = parser.parse_args()
	main()