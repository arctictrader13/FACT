
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
import matplotlib.pyplot as plt

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
data_PATH= PATH + 'dataset/'

batch_size = 6
max_train_steps = 1
initial_learning_rate = 0.001
lr_decresing_step = 3
lr_divisor = 10


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
#device = "cpu"

def train(data_loader, model, max_train_steps, use_saliency=False, \
          k_most_salient=0, saliency_method=None, saliency_method_name=None, \
          initial_model=None):
    learning_rate = initial_learning_rate
    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    accuracies = []
    losses = []
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs = batch_inputs.requires_grad_()
        if use_saliency:
            _ = initial_model.forward(batch_inputs)
            saliency_map = saliency_method.saliency(batch_inputs)
            data = remove_salient_pixels(batch_inputs, saliency_map)
        else:
            data = batch_inputs

        output = model.forward(data)

        loss = criterion(output, batch_targets)
        accuracy = float(sum(batch_targets == torch.argmax(output, 1))) / float(batch_size)

        # conpute gradients
        loss.backward()

        # update weights
        optimizer.step()

        if step % lr_decresing_step == 0:
            learning_rate /= lr_divisor
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
            break
    # plot
    plt.plot(range(len(losses)), losses)
    plt.ylabel('Loss')
    plt.xlabel('Batches')
    if use_saliency:
        figname = saliency_method_name + "_" + str(k_most_salient) + ".jpeg"
    else:
        figname = "initial_model.jpeg"
    plt.savefig(os.path.join("results", "remove_and_retrain", figname))
    return accuracies[-1]


def remove_and_retrain(data_loader):
    initial_model = resnet18(pretrained=True)
    initial_accuracy = train(data_loader, initial_model, max_train_steps, \
                             use_saliency=False)
    saliency_methods = [(FullGrad(initial_model), "FullGrad"), 
                       (SimpleFullGrad(initial_model), "SimpleFullGrad")]
    percentages = [0.1, 0.2, 0.5, 0.8, 1.0, 2, 5, 10]
    ks = [int(i * 224 * 224 / 100) for i in percentages] 
    accuracies = torch.zeros((len(saliency_methods), len(ks)))
    for method_idx, (saliency_method, method_name) in enumerate(saliency_methods):
        for k_idx, k in enumerate(ks):
            print("Run saliency method: ", method_name)

            model = resnet18(pretrained=True)
            accuracy = train(data_loader, model, max_train_steps, \
                             use_saliency=True, k_most_salient=k, \
                             saliency_method=saliency_method, \
                             saliency_method_name=method_name, \
                             initial_model=initial_model)
            accuracies[method_idx, k_idx] = accuracy
        print(percentages, accuracies[method_idx])
        plt.plot(percentages, list(accuracies[method_idx]), label="method_name" + str(k))
    plt.ylabel('Accuracy')
    plt.xlabel('k %')
    plt.savefig(os.path.join("results", "remove_and_retrain", "final_result.jpeg"))


def main():
    # same transformations for each dataset
    transform_standard = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]), ])
    dataset = data_PATH + "/cifar/"
    data = datasets.CIFAR100(dataset, train=True, transform=transform_standard, \
        target_transform=None, download=True)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

    remove_and_retrain(data_loader)


if __name__ == "__main__":
    main()
