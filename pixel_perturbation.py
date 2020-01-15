
import torch
from torchvision import datasets, transforms, utils
import os

# Import saliency methods and models
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from models.vgg import *
from models.resnet import *
from misc_functions import *

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
dataset = PATH + 'dataset/'

batch_size = 1
k = 100

device = "cpu"

def main():
	# Dataset loader for sample images
	sample_loader = torch.utils.data.DataLoader(
		datasets.ImageFolder(dataset, transform=transforms.Compose([
						   transforms.Resize((224,224)),
						   transforms.ToTensor(),
						   transforms.Normalize(mean = [0.485, 0.456, 0.406],
											std = [0.229, 0.224, 0.225])
					   ])),
		batch_size= batch_size, shuffle=False)

	unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
							   std = [0.229, 0.224, 0.225])


	# uncomment to use VGG
	# model = vgg16_bn(pretrained=True)
	model = resnet18(pretrained=True)

	# Initialize FullGrad objects
	fullgrad = FullGrad(model)
	simple_fullgrad = SimpleFullGrad(model)

	save_path = PATH + 'results/'

	for batch_idx, (data, target) in enumerate(sample_loader):
		data, target = data.to(device).requires_grad_(), target.to(device)

		# Run Input through network
		initial_output = model.forward(data)
		
		# Compute saliency maps for the input data
		cam = fullgrad.saliency(data)
		cam_simple = simple_fullgrad.saliency(data)
		
		[column_size, row_size] = data.size()[2:4]
		for index in torch.topk(cam.view((-1)), k=3, largest=False)[1]:
			row = index / row_size
			column = index % row_size 
			data[0, 0, row, column] = 0.0

		final_output = model.forward(data)
		print(initial_output.size(), final_output.size())
		print((initial_output - final_output).abs_().sum())

if __name__ == "__main__":
	main()