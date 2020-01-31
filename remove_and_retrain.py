
import os

import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
import torch
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

# Import saliency methods and models
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from saliency.inputgradient import Inputgrad
from models.vgg import vgg11
import misc_functions

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(model, config, data_loader=None, data_path="", plot_name=""):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.classifier.parameters(),
                                lr=config['initial_learning_rate'], momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size=config['lr_decresing_step'],
                                    gamma=config['lr_gamma'])
    model.train()

    if data_loader != None:
        data_iterator = iter(data_loader)

    losses = []
    loss_steps = []
    num_batches = 0.0
    for _ in range(config['epochs']):
        accuracy = 0.0
        step = 0
        while step != config['max_train_steps']:
            if data_path != "":
                if not os.path.exists(os.path.join(data_path,
                                                   "batch_input" + str(step))):
                    break
                batch_inputs = torch.load(
                    os.path.join(data_path, "batch_input" + str(step)))
                batch_targets = torch.load(
                    os.path.join(data_path, "batch_target" + str(step)))
            else:
                try:
                    batch_inputs, batch_targets = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(data_loader)
                    break
            batch_inputs = batch_inputs.to(config['device'])
            batch_targets = batch_targets.to(config['device'])

            num_batches += 1

            optimizer.zero_grad()
            output = model.forward(batch_inputs)

            loss = criterion(output, batch_targets)
            accuracy += sum(batch_targets == torch.argmax(output, 1))

            loss.backward()
            optimizer.step()

            if step % config['print_step'] == 0 and step != 0:
                losses += [float(loss)]
                loss_steps += [num_batches]
                accuracy = float(accuracy) / float(config['batch_size'] * config['print_step'])

                if (len(losses) > 2 and abs(losses[-1] - losses[-2]) < 0.0001):
                    break
                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, "
                      "Accuracy = {:.2f}, Train Loss = {:.3f}".format(
                          datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                          0, config['batch_size'], accuracy, loss
                          ))
                accuracy = 0.0
            step += 1

        scheduler.step()

    # plot
    plt.figure(1)
    plt.clf()
    plt.plot(loss_steps, losses)
    plt.ylabel('Loss')
    plt.xlabel('Batches')
    plt.savefig(os.path.join("results", "remove_and_retrain",
                             "plots", plot_name + ".jpg"))

    torch.save(model, os.path.join("results", "remove_and_retrain",
                                   "models", plot_name))

    with open(os.path.join("results", "remove_and_retrain",
                           "training_output", plot_name + ".json"), 'w') as f:
        json.dump({"steps": loss_steps, "loss": losses}, f)


def test(model, config, data_loader=None, data_path=""):
    model.eval()
    accuracies = []
    step = 0

    if data_loader != None:
        data_iterator = iter(data_loader)

    while step != config['max_train_steps']:
        if data_path != "":
            if not os.path.exists(os.path.join(data_path,
                                               "batch_input" + str(step))):
                break
            batch_inputs = torch.load(
                os.path.join(data_path, "batch_input" + str(step)))
            batch_targets = torch.load(
                os.path.join(data_path, "batch_target" + str(step)))
        else:
            try:
                batch_inputs, batch_targets = next(data_iterator)
            except StopIteration:
                break
        batch_inputs = batch_inputs.to(config['device'])
        batch_targets = batch_targets.to(config['device'])
        batch_inputs.requires_grad = False

        output = model.forward(batch_inputs)
        accuracies += [float(sum(batch_targets ==
                                 torch.argmax(output, 1))) / config['batch_size']]
        step += 1
    accuracies = torch.tensor(accuracies)
    return float(accuracies.mean()), float(accuracies.std())


def init_model(config):
    model = vgg11(pretrained=True, in_size=32).to(config['device'])
    for param in  model.features:
        param.requires_grad = False

    model.classifier[0] = torch.nn.Linear(in_features=512, out_features=256, bias=True).to(config['device'])
    model.classifier[3] = torch.nn.Linear(in_features=256, out_features=128, bias=True).to(config['device'])
    model.classifier[6] = torch.nn.Linear(in_features=128, out_features=10, bias=True).to(config['device'])

    return model


def get_saliency_methods(grad_names, initial_model):
    saliency_methods = []
    for grad_name in grad_names:
        if grad_name == "fullgrad":
            saliency_methods += [(FullGrad(initial_model, im_size=(3, 32, 32)), "FullGrad")]
        elif grad_name == "simplegrad":
            saliency_methods += [(SimpleFullGrad(initial_model), "SimpleFullGrad")]
        elif grad_name == "inputgrad":
            saliency_methods += [(Inputgrad(initial_model), "Input-Gradient")]
        elif grad_name == "random":
            saliency_methods += [(None, "Random")]
    return saliency_methods


def remove_and_retrain(data_path, config, addition=""):
    initial_model = torch.load(os.path.join("results", "remove_and_retrain", "models", "initial_model"))
    saliency_methods = get_saliency_methods(config['grads'], initial_model)

    colors = {"FullGrad": "#0074d9", "gradcam": "#111111",
              "Random": "#f012be", "Input-Gradient": "#01ff70"}
    total_features = config['img_size'] * config['img_size']
    accuracies_mean = [[0.0 for _ in range(len(config['k']))] for _ in range(len(saliency_methods))]
    accuracies_std = [[0.0 for _ in range(len(config['k']))] for _ in range(len(saliency_methods))]
    for method_idx, (_, method_name) in enumerate(saliency_methods):
        modified_data_path = os.path.join(data_path, "modified_cifar_10", method_name)

        for k_idx, k in enumerate(config['k']):
            print("Run saliency method: ", method_name)
            train_data_path = os.path.join(modified_data_path, str(int(k * total_features)))
            print(train_data_path)
            model = init_model(config)
            train(model, config, data_path=os.path.join(train_data_path, "train"),
                  plot_name=method_name + "_" + str(k))
            accuracies_mean[method_idx][k_idx], accuracies_std[method_idx][k_idx] \
             = test(model, config, data_path=os.path.join(train_data_path, "test"))

        plt.figure(0)
        print([k * 100 for k in config['k']])
        print(accuracies_mean[method_idx])
        print(accuracies_std[method_idx])
        plt.errorbar([k * 100 for k in config['k']], accuracies_mean[method_idx],
                     accuracies_std[method_idx], label=method_name,
                     color=colors[method_name])

    plt.figure(0)
    plt.ylabel('Accuracy')
    plt.xlabel('k %')
    plt.legend()
    plt.savefig(os.path.join("results", "remove_and_retrain", addition + "final_plot.jpeg"))

    with open(os.path.join("results", "remove_and_retrain",
                           addition + "final_output" + ".json"), 'w') as f:
        json.dump({"mean": accuracies_mean, "std": accuracies_std}, f)


def compute_modified_datasets(train_set_loader, test_set_loader, data_path, config):
    initial_model = torch.load(os.path.join("results", "remove_and_retrain",
                                            "models", "initial_model"))
    saliency_methods = get_saliency_methods(config['grads'], initial_model)

    total_features = config['img_size'] * config['img_size']
    for _, (saliency_method, method_name) in enumerate(saliency_methods):
        for dataset, dataloader in [("train", train_set_loader), ("test", test_set_loader)]:
            saliency_path = os.path.join(data_path, "saliency_maps", method_name + "_vgg11", dataset)
            if saliency_method != None:
                misc_functions.compute_and_store_saliency_maps(dataloader, initial_model, \
                    config['device'], config['max_train_steps'], saliency_method, saliency_path)
            for k in config['k']:
                num_pixels = int(k * total_features)

                dataset_path = os.path.join(data_path, "modified_cifar_10", method_name, str(num_pixels))
                misc_functions.create_folder(os.path.join(dataset_path, dataset))

                for step, (batch_inputs, batch_targets) in enumerate(dataloader):
                    if method_name == "Random":
                        data = misc_functions.remove_random_salient_pixels(batch_inputs, 42, k, im_size=32)
                    else:
                        saliency_map = torch.load(os.path.join(saliency_path, \
                            "saliency_map_" + str(step)))
                        data = misc_functions.remove_salient_pixels(batch_inputs, saliency_map, \
                            num_pixels=num_pixels, most_salient=config['most_salient'])
                    torch.save(data, os.path.join(dataset_path, dataset, "batch_input" + str(step)))
                    torch.save(batch_targets, os.path.join(dataset_path, dataset, "batch_target" + str(step)))

                    if step == config['max_train_steps']:
                        break


def load_cifar_dataset(data_path, config):
    transform_standard = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]), ])
    cifar_path = os.path.join(data_path, "cifar")
    train_set = datasets.CIFAR10(cifar_path, train=True, transform=transform_standard, \
        target_transform=None, download=True)
    test_set = datasets.CIFAR10(cifar_path, train=False, transform=transform_standard, \
        target_transform=None, download=True)
    train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=False)
    test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

    return train_set_loader, test_set_loader


def main():
    config = vars(ARGS)
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config['data_directory'])

    if config['phase'] == "train_initial_model":
        train_set_loader, test_set_loader = load_cifar_dataset(data_path, config)
        initial_model = init_model(config)

        train(initial_model, config, data_loader=train_set_loader, plot_name="initial_model")

        initial_accuracy_mean, initial_accuracy_std = test(initial_model, config, data_loader=test_set_loader)
        print(initial_accuracy_mean, initial_accuracy_std)

    elif config['phase'] == "create_modified_datasets":
        train_set_loader, test_set_loader = load_cifar_dataset(data_path, config)
        compute_modified_datasets(train_set_loader, test_set_loader, data_path, config)

    elif config['phase'] == "train_on_modified_datasets":
        remove_and_retrain(data_path, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=10, type=int,
                        help='Number of images passed through at once')
    parser.add_argument('--data_directory', default="dataset", type=str,
                        help='Name of the directory containing the datasets')
    parser.add_argument('--device', default="cuda:0", type=str,
                        help='cpu or gpu')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Maximum number of epochs')
    parser.add_argument('--grads', default=["fullgrad"], type=str, nargs='+',
                        help='which grad methods to be applied')
    parser.add_argument('--img_size', default=32, type=int,
                        help='Row and Column size of the image')
    parser.add_argument('--initial_learning_rate', default=0.001, type=float,
                        help='Initial learning rate')
    parser.add_argument('--k', default=[0.1, 0.25, 0.5, 0.75, 0.9],
                        type=float,nargs="+",
                        help='Percentage of k% most salient pixels')
    parser.add_argument('--lr_decresing_step', default=1, type=int,
                        help='Number of training steps between decreasing the \
                        learning rate')
    parser.add_argument('--lr_gamma', default=0.1, type=float,
                        help='mltiplier for changing the lr')
    parser.add_argument('--max_train_steps', default=-1, type=int,
                        help='Maximum number of training steps')
    parser.add_argument('--most_salient', default="True", type=str,
                        help='most salient = True or False depending on retrain or pixel perturbation')
    parser.add_argument('--n_random_runs', default=5, type=int,
                        help='Number of runs for random pixels to be removed to decrease std of random run')
    parser.add_argument('--phase', default="train_initial_model", type=str,
                        help='Slect phase of the experiment. Options: train_initial_model, \
                                create_modified_datasets, train_on_modified_datasets')
    parser.add_argument('--print_step', default=500, type=int,
                        help='Number of batches after which we print')
    parser.add_argument('--replacement', default="black", type=str,
                        help='black = 1.0 or mean = [0.485, 0.456, 0.406]')
    parser.add_argument('--target_layer', default="layer4", type=str,
                        help='Which layer to be visualized in GRADCAM')


    ARGS = parser.parse_args()

    main()
