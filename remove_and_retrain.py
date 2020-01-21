
import os

import torch
from torchvision import datasets, transforms
from datetime import datetime

# Import saliency methods and models
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from models.vgg import vgg16_bn
from models.resnet import resnet18
from misc_functions import create_folder, compute_and_store_saliency_maps, remove_salient_pixels
import copy

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
data_PATH= PATH + 'dataset/'

batch_size = 3
k_most_salient = 100
max_train_steps = 3
learning_rate = 0.01

device = "cpu"

def train(data_loader, model, max_train_steps, learning_rate, use_saliency=False, saliency_path=""):
    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.RMSprop(model.parameters(), \
                                    lr=learning_rate)
    accuracies = []
    losses = []
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        if use_saliency:
            saliency_map = torch.load(os.path.join(saliency_path, "full", "saliency_map_" + str(step)))
            data = remove_salient_pixels(batch_inputs, saliency_map)
            #import pdb; pdb.set_trace()
        else:
            data = copy.copy(batch_inputs)

        output = model.forward(data)

        loss = criterion(output, batch_targets)
        accuracy = float(sum(batch_targets == torch.argmax(output, 1))) / float(batch_size)

        # conpute gradients
        loss.backward()

        # update weights
        optimizer.step()

        if step % 1 == 0:
            accuracies += [accuracy]
            losses += [loss]
            if (len(losses) > 2 and abs(losses[-1] - losses[-2]) < 0.0001):
                break
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    max_train_steps, batch_size, accuracy, loss
            ))

        if step == max_train_steps:
            print("break")
            break


def remove_and_retrain(data_loader, model):
    model = resnet18(pretrained=True)

    train(data_loader, model, max_train_steps, learning_rate, use_saliency=False)
    saliency_path = os.path.join(data_PATH, "saliency_maps", "cifar_resnet18")
    if not os.path.isdir(saliency_path):
        create_folder(saliency_path)
        compute_and_store_saliency_maps(data_loader, model, device, saliency_path)
    # save model
    model = resnet18(pretrained=True)
    train(data_loader, model, max_train_steps, learning_rate, use_saliency=True, saliency_path=saliency_path)


def main():
    # same transformations for each dataset
    transform_standard = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]), ])
    dataset = data_PATH + "/cifar/"
    data = datasets.CIFAR100(dataset, train=True, transform=transform_standard,target_transform=None,
                                         download=True)
    # Dataset loader for sample images
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    
    # # Dataset loader for sample images
    # data_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(dataset, transform=transform_standard),
    #     batch_size=batch_size, shuffle=False)

    # model = vgg16_bn(pretrained=True)
    model = resnet18(pretrained=True)
        
    remove_and_retrain(data_loader, model)


if __name__ == "__main__":
    main()
