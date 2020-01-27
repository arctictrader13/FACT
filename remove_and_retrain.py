
import os

import torch
from torchvision import datasets, transforms
from datetime import datetime

# Import saliency methods and models
import argparse
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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(data_loader, model, use_saliency=False, \
          k_most_salient=0, saliency_method=None, saliency_method_name=None):
    learning_rate = ARGS.initial_learning_rate
    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    accuracies = []
    losses = []
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs, batch_targets = batch_inputs.to(ARGS.device), batch_targets.to(ARGS.device)
        if use_saliency:
            batch_inputs = batch_inputs.detach()
            saliency_map = saliency_method.saliency(batch_inputs)
            data = remove_salient_pixels(batch_inputs, saliency_map, num_pixels=k_most_salient, most_salient=ARGS.most_salient)
        else:
            data = batch_inputs
        
        data.requires_grad = True
        data = data.to(ARGS.device)
        batch_targets = batch_targets.to(ARGS.device)
        
        output = model.forward(data)
        loss = criterion(output, batch_targets)
        accuracy = float(sum(batch_targets == torch.argmax(output, 1))) / float(ARGS.batch_size)

        # conpute gradients
        loss.backward()

        # update weights
        optimizer.step()

        if step % ARGS.lr_decresing_step == 0:
            learning_rate /= ARGS.lr_divisor
            accuracies += [accuracy]
            losses += [loss]
            if (len(losses) > 2 and abs(losses[-1] - losses[-2]) < 0.0001):
                break
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    ARGS.max_train_steps, ARGS.batch_size, accuracy, loss
            ))

        if step == ARGS.max_train_steps:
            break
    # plot
    plt.figure(1)
    plt.clf()
    plt.plot(range(len(losses)), losses)
    plt.ylabel('Loss')
    plt.xlabel('Batches')
    if use_saliency:
        figname = saliency_method_name + "_" + str(k_most_salient) + ".jpeg"
    else:
        figname = "initial_model.jpeg"
    plt.savefig(os.path.join("results", "remove_and_retrain", figname))
    return accuracies[-1]


def get_cifar_ready_resnet():
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 100)
    model = model.to(ARGS.device)
    return model


def remove_and_retrain(data_loader):
    initial_model = get_cifar_ready_resnet()
    initial_accuracy = train(data_loader, initial_model, use_saliency=False)
    saliency_methods = []
    print(next(initial_model.parameters()).device)
    for grad_name in ARGS.grads:
        if grad_name == "fullgrad":
            saliency_methods += [(FullGrad(initial_model), "FullGrad")]
        elif grad_name == "simplegrad":
            saliency_methods += [(SimpleFullGrad(initial_model), "SimpleFullGrad")]
        # TODO add gradcam
    total_features = 224 * 224
    accuracies = torch.zeros((len(saliency_methods), len(ARGS.k)))
    for method_idx, (saliency_method, method_name) in enumerate(saliency_methods):
        for k_idx, k in enumerate(ARGS.k):
            print("Run saliency method: ", method_name)

            model = get_cifar_ready_resnet()
            accuracy = train(data_loader, model, use_saliency=True, \
                             k_most_salient=int(k * total_features), \
                             saliency_method=saliency_method, \
                             saliency_method_name=method_name)
            accuracies[method_idx, k_idx] = accuracy
        plt.figure(0)
        plt.plot(ARGS.k, list(accuracies[method_idx]), label=method_name + str(k))
    plt.figure(0)
    plt.ylabel('Accuracy')
    plt.xlabel('k %')
    plt.legend()
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
    data_loader = torch.utils.data.DataLoader(data, batch_size=ARGS.batch_size, shuffle=False)

    remove_and_retrain(data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', default=[0.001, 0.005, 0.01, 0.05, 0.1], type=float,nargs="+",
                        help='Percentage of k% most salient pixels')
    parser.add_argument('--most_salient', default=True, type=bool,
                        help='most salient = True or False depending on retrain or pixel perturbation')
    parser.add_argument('--grads', default=["fullgrad"], type=str, nargs='+',
                        help='which grad methods to be applied')
    parser.add_argument('--device', default="cuda:0", type=str,
                        help='cpu or gpu')
    parser.add_argument('--target_layer', default="layer4", type=str,
                        help='Which layer to be visualized in GRADCAM')
    parser.add_argument('--n_random_runs', default=5, type=int,
                        help='Number of runs for random pixels to be removed to decrease std of random run')
    parser.add_argument('--replacement', default="black", type=str,
                        help='black = 1.0 or mean = [0.485, 0.456, 0.406]')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Number of images passed through at once')
    parser.add_argument('--max_train_steps', default=100, type=int,
                        help='Maximum number of training steps')
    parser.add_argument('--initial_learning_rate', default=0.01, type=float,
                        help='Initial learning rate')
    parser.add_argument('--lr_decresing_step', default=10, type=int,
                        help='Number of training steps between decreasing the learning rate')
    parser.add_argument('--lr_divisor', default=10, type=float,
                        help='Divisor used to decrease the leaarning rate')

    ARGS = parser.parse_args()
    main()


