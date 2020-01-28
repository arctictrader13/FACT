#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Compute saliency maps of images from dataset folder
    and dump them in a results folder """

import torch
from torchvision import datasets, transforms, utils
import os

# Import saliency methods and models
from saliency.inputgradient import Inputgrad
from pixel_perturbation import *
from models.vgg import *
from models.resnet import *
from gradcam import grad_cam
from misc_functions import *

batch_size = 10
salient_type = "most"
grad_type = "inputgrad"

# PATH variables
# PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
PATH = os.path.abspath(os.getcwd()) + "/"
dataset = PATH + 'dataset/'
result_path = PATH + 'results/imagenet/'

# cuda = torch.cuda.is_available()
# device = torch.device("cuda" if cuda else "cpu")
device = "cpu"

data_PATH = PATH + 'dataset/'
dataset = data_PATH

sample_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(dataset, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])),
    batch_size=batch_size, shuffle=False)

unnormalize = NormalizeInverse(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])

# uncomment to use VGG
model_name = "resnet18" # or "vgg16_bn
target_layer = "layer4" # for vgg: features
grads = ["inputgrad", "fullgrad", "gradcam"]
n_images_save = 50

save_path = PATH + 'results/'


def get_filename(result_path, grad_type, index):
    filename = result_path + "/" + grad_type + "_" + model_name + "_" + str(index) + ".png"
    return filename


def compute_saliency_and_save():
    inputgrad_bool = False
    for grad_type in grads:
        model, grad = init_grad_and_model(grad_type, model_name, device)

        if grad_type == "inputgrad":
            inputgrad_bool = True

        grad_counter = 0
        print("grad_type:{}".format(grad_type))

        # print("grad:{}".format(grad))
        grad_counter += 1
        counter = 1

        for batch_idx, (data, target) in enumerate(sample_loader):
            if counter >= n_images_save:
                break
            data, target = data.to(device).requires_grad_(), target.to(device)
            # data, _ = next(iter(sample_loader))

            # Compute saliency maps for the input data
            if grad_type == "gradcam":
                probs, ids = grad.forward(data)
                # Grad-CAM
                grad.backward(ids=ids[:, [0]])
                saliency = grad.generate(target_layer=target_layer)

            else:
                saliency = grad.saliency(data)

            for i in range(len(data)):
                im = unnormalize(data[i, :, :, :].cpu())
                im = im.view(1, 3, 224, 224)[-1, :, :, :]
                reg = saliency[i, :, :, :]
                filename = get_filename(result_path, grad_type, counter)
                counter += 1
                print(filename)

                # print("filename:{}".format(filename))
                save_saliency_map_inputgrad(im, reg, filename, inputgrad=inputgrad_bool)


def save_saliency_map_inputgrad(image, saliency_map, filename, inputgrad=False):
    """
    Save saliency map on image.

    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W)
        filename: string with complete path and file extension

    """
    image = image.data.cpu().numpy()

    saliency_map = saliency_map.data.cpu().numpy()

    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = saliency_map.clip(0, 1)

    saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
    saliency_map = cv2.resize(saliency_map, (224, 224))

    image = np.uint8(image * 255).transpose(1, 2, 0)
    image = cv2.resize(image, (224, 224))
    # Apply JET colormap
    color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)

    # Combine image with heatmap
    if inputgrad is True:
        img_with_heatmap = np.float32(color_heatmap)
    else:
        img_with_heatmap = np.float32(color_heatmap) + np.float32(image)

    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)

    cv2.imwrite(filename, np.uint8(255 * img_with_heatmap))

if __name__ == "__main__":
    # Create folder to saliency maps
    compute_saliency_and_save()
    print('Saliency maps saved.')
