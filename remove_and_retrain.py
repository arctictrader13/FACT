
import os

import torch
from torchvision import datasets, transforms
from datetime import datetime

# Import saliency methods and models
import argparse
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from models.vgg import vgg16_bn, vgg11
from models.resnet import resnet18
from misc_functions import create_folder, compute_and_store_saliency_maps, remove_salient_pixels
import copy
import matplotlib.pyplot as plt

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
data_PATH= PATH + 'dataset/'

#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.enabled = False

def train(data_loader, model, k_most_salient=0, saliency_path="", saliency_method_name=""):
    learning_rate = ARGS.initial_learning_rate
    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    
    accuracies = []
    losses = []
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs, batch_targets = batch_inputs.to(ARGS.device), \
                                      batch_targets.to(ARGS.device)
        if k_most_salient != 0:
            saliency_map = torch.load(os.path.join(saliency_path, \
                    "saliency_map_" + str(step)))
            data = remove_salient_pixels(batch_inputs, saliency_map, \
                    num_pixels=k_most_salient, most_salient=ARGS.most_salient)
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
    plt.plot(range(0, ARGS.batch_size * len(losses), ARGS.batch_size), losses)
    plt.ylabel('Loss')
    plt.xlabel('Batches')
    if k_most_salient != 0:
        figname = saliency_method_name + "_" + str(k_most_salient) + ".jpeg"
    else:
        figname = "initial_model.jpeg"
    plt.savefig(os.path.join("results", "remove_and_retrain", figname))


def test(data_loader, model, max_steps, k_most_salient=0, saliency_path=""):
    loss = 0.0
    accuracy = 0.0
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs, batch_targets = batch_inputs.to(ARGS.device), \
                                      batch_targets.to(ARGS.device)
        if k_most_salient != 0: 
            saliency_map = torch.load(os.path.join(saliency_path, \
                    "saliency_map_" + str(step)))
            data = remove_salient_pixels(batch_inputs, saliency_map, \
                    num_pixels=k_most_salient, most_salient=ARGS.most_salient)
        else:
            data = batch_inputs
        batch_inputs.requires_grad = False
        batch_inputs = batch_inputs.to(ARGS.device).detach()
        batch_targets = batch_targets.to(ARGS.device)
        
        output = model.forward(batch_inputs)
        accuracy += float(sum(batch_targets == torch.argmax(output, 1))) / float(ARGS.batch_size)
        if step == max_steps:
            break

    # TODO check if correct
    num_batches = len(data_loader.dataset)
    return loss / num_batches, accuracy / num_batches


def remove_and_retrain(train_set_loader, test_set_loader):
    initial_model = vgg11(pretrained=True).to(ARGS.device)
    train(train_set_loader, initial_model)
    initial_loss, initial_accuracy = test(test_set_loader, initial_model, ARGS.max_train_steps)
    
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
        train_saliency_path = os.path.join(data_PATH, "saliency_maps", method_name + "_vgg11", "train")
        compute_and_store_saliency_maps(train_set_loader, initial_model, \
            ARGS.device, ARGS.max_train_steps, saliency_method, train_saliency_path)

        test_saliency_path = os.path.join(data_PATH, "saliency_maps", method_name + "_vgg11", "test")
        compute_and_store_saliency_maps(test_set_loader, initial_model, \
            ARGS.device, ARGS.max_train_steps, saliency_method, test_saliency_path)

        for k_idx, k in enumerate(ARGS.k):
            print("Run saliency method: ", method_name)

            model = vgg11(pretrained=True).to(ARGS.device)
            train(train_set_loader, model, \
                  k_most_salient=int(k * total_features), \
                  saliency_path=train_saliency_path, saliency_method_name=method_name)
            loss, accuracy = test(test_set_loader, model, \
                            ARGS.max_train_steps, \
                            k_most_salient=int(k * total_features), \
                            saliency_path=test_saliency_path)
            accuracies[method_idx, k_idx] = accuracy
        plt.figure(0)
        plt.plot([k * 100 for k in ARGS.k], list(accuracies[method_idx]), label=method_name + str(k))
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
    train_set = datasets.CIFAR100(dataset, train=True, transform=transform_standard, \
        target_transform=None, download=True)
    test_set = datasets.CIFAR100(dataset, train=False, transform=transform_standard, \
        target_transform=None, download=True)
    train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=ARGS.batch_size, shuffle=False)
    test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=ARGS.batch_size, shuffle=False)

    remove_and_retrain(train_set_loader, test_set_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', default=[0.001, 0.005, 0.01, 0.05, 0.1], type=float,nargs="+",
                        help='Percentage of k% most salient pixels')
    parser.add_argument('--most_salient', default="True", type=str,
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
    parser.add_argument('--initial_learning_rate', default=0.0001, type=float,
                        help='Initial learning rate')
    parser.add_argument('--lr_decresing_step', default=10, type=int,
                        help='Number of training steps between decreasing the learning rate')
    parser.add_argument('--lr_divisor', default=10, type=float,
                        help='Divisor used to decrease the leaarning rate')

    ARGS = parser.parse_args()
    main()


