
import os

import torch
from torchvision import datasets, transforms

# Import saliency methods and models
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from models.vgg import vgg16_bn
from models.resnet import resnet18
from misc_functions import create_folder, compute_and_store_saliency_maps


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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])),
        batch_size=batch_size, shuffle=False)

    # model = vgg16_bn(pretrained=True)
    model = resnet18(pretrained=True)

    saliency_path = os.path.join(dataset, "saliency_maps", "imagenet_resnet18")

    if not os.path.isdir(saliency_path):
        create_folder(saliency_path)
        compute_and_store_saliency_maps(sample_loader, model, device, saliency_path)


if __name__ == "__main__":
    main()
