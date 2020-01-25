import torch
from torchvision import datasets, transforms, utils, models
import os

# Import saliency methods and models
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from models.vgg import *
from models.resnet import *
from misc_functions import *
from gradcam import grad_cam
from gradcam import main

import argparse
import torchvision
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn.functional as F

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'

# console testing
# PATH = os.path.abspath(os.getcwd()) + "/"

data_PATH = PATH + 'dataset/'

result_path = PATH + 'results/'

unnormalize = NormalizeInverse(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    device = ARGS.device

    batch_size = ARGS.batch_size

    if ARGS.replacement == "mean":
        replacement = [0.485, 0.456, 0.406]
    else:
        replacement = [1.0]

    if ARGS.most_salient is True:
        salient_type = "most"
    else:
        salient_type = "least"

    # same transformations for each dataset
    transform_standard = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]), ])

    if ARGS.dataset == "cifar":
        dataset = data_PATH + "/cifar/"
        data = torchvision.datasets.CIFAR100(dataset, train=True, transform=transform_standard, target_transform=None,
                                             download=True)
        # Dataset loader for sample images
        sample_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    elif ARGS.dataset == "mnist":
        dataset = data_PATH + "/mnist/"
        data = torchvision.datasets.MNIST(dataset, train=True, transform=transform_standard, target_transform=None,
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

    model_name = ARGS.model + ARGS.model_type

    # simple fullgrad
    # simple_fullgrad = SimpleFullGrad(model)

    save_path = PATH + 'results/' + ARGS.dataset

    # initialize results dictionary: key: gradient method (random, fullgrad,...), values: [[mean, std],..] per k%
    results_dict = {}
    all_results = [{},{},{}]
    total_features = 224 * 224

    grad = None
    model = None

    for grad_type in ARGS.grads:
        model, grad = init_grad_and_model(grad_type, model_name, device)

        grad_counter = 0
        score_means, score_stds, prob_means, prob_stds, kl_divs = [], [], [], [], []

        print("grad_type:{}".format(grad_type))

        for i in ARGS.k:

            # print("grad:{}".format(grad))
            grad_counter +=1
            k_most_salient = int(i * total_features)
            # print("k_most_salient:{}".format(k_most_salient))
            counter = 0
            tmp_results = [[], [], []] # score diffs, prob diffs, kl divs

            for batch_idx, (data, target) in enumerate(sample_loader):
                counter += 1

                # console testing
                # data, target = next(iter(sample_loader))

                # for debugging purposes
                if counter % 100 == 0:
                    print("{} image batches processed".format(counter))
                if counter == ARGS.n_images:
                    break

                data = data.to(device).requires_grad_()

                # Run Input through network (two different networks if gradcam or fullgrad)
                with torch.no_grad():
                   initial_output = model.forward(data)

                # compute saliency maps for grad methods not random
                if grad_type != "random":

                    # print("data size:{}".format(data.size()))
                    cam = compute_saliency_per_grad(grad_type, grad, data)
                    data = data.detach()
                    #print(cam)
                    #print("cam size:{}".format(cam.size()))
                    if ARGS.save_grad is True and grad_counter == 1 and counter <= ARGS.n_save:
                        save_saliency_map_batch(cam, data, result_path, grad_type, salient_type, counter)

                    data = remove_salient_pixels(data, cam, num_pixels=k_most_salient, most_salient=ARGS.most_salient,
                                                 replacement=replacement)

                    tmp_results = abs_frac_per_grad(model, data, initial_output, tmp_results)

                # change pixels based on random removal
                elif grad_type == "random":
                    data = data.detach()
                    # run n_random_runs for random pixel removal
                    sample_seeds = np.random.randint(0, 10000, ARGS.n_random_runs)
                    for seed in sample_seeds:
                        tmp_data = remove_random_salient_pixels(data, seed, k_percentage=i, replacement=replacement)
                        tmp_results = abs_frac_per_grad(model, tmp_data, initial_output, tmp_results)

                # print("counter:{}".format(counter))
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            # print(torch.cuda.memory_allocated(device=None))
            #print("Absolute fractional output changes: ", tmp_results)
            # print("Actual values: ",  initial_class_probability, final_class_probability)
            # save mean and std of
            append_mean_std(tmp_results[0], score_means, score_stds)
            append_mean_std(tmp_results[1], prob_means, prob_stds)
            kl_divs.append(tmp_results[2])


            # torch.cuda.empty_cache()

        all_results[0][grad_type] = [score_means, score_stds]
        all_results[1][grad_type] = [prob_means, prob_stds]
        all_results[2][grad_type] = [kl_divs]

    # plot for all gradient methods stds and means for all k% values
    save_experiment_file = result_path + ARGS.dataset + "_" + model_name + "_" + salient_type + "_" + ARGS.replacement + str(ARGS.n_images) + "_"

    # plot score differences, prob differences and kl divergence
    plot_all_grads(all_results[0], filename=save_experiment_file + "scores")
    plot_all_grads(all_results[1], filename=save_experiment_file + "probs")
    plot_all_grads(all_results[2], filename=save_experiment_file + "kl", div=True)

    print("For K values: {}".format(ARGS.k))
    print("Score absolute fractional differences: {}".format(all_results[0]))
    print("Probs absolute fractional differences: {}".format(all_results[1]))
    print("KL divergences per k: {}".format(all_results[0]))


def abs_frac_per_grad(model, data, initial_output, tmp_results):
    # output after pixel perturbation
    with torch.no_grad():
        final_output = model.forward(data)

    final_output.to("cpu")
    initial_output.to("cpu")

    # FOR KL DIVERGENCE AND OWN METRIC
    # get probabilities instead of output scores
    initial_probabilities = F.softmax(initial_output, dim=1)
    final_probabilities = F.softmax(final_output, dim=1)

    kl_div = F.kl_div(final_probabilities,initial_probabilities, reduction='batchmean')

    # score and prob differences
    tmp_score_diffs = get_temp_result(initial_output, final_output)
    tmp_prob_diffs = get_temp_result(initial_probabilities, final_probabilities)

    # save per image
    tmp_results[0].append(np.round(tmp_score_diffs.tolist(), 8))
    tmp_results[1].append(np.round(tmp_prob_diffs.tolist(), 8))
    tmp_results[2].append(kl_div.item())

    return tmp_results

def init_grad_and_model(grad_type, model_name, device):
    if grad_type == 'gradcam':
        # Gradcam
        model, grad = initialize_grad_cam(model_name, device)
    else:
        model = eval(model_name)(pretrained=True)
        model = model.to(device)
        model.eval()
        # Initialize Gradient objects
        grad = FullGrad(model)

    return model, grad

def get_temp_result(initial_output, final_output):

    # initially most confident class
    initial_class_scores, predicted_class = initial_output.max(1)
    # same value after modification
    final_class_scores = final_output.index_select(1, predicted_class).max(0)[0]

    # absolute fractional difference of raw results
    tmp_result = abs(final_class_scores - initial_class_scores) / initial_class_scores

    return tmp_result

def append_mean_std(tmp_results, means, stds):
    means.append(np.mean(tmp_results))
    stds.append(np.std(tmp_results))


def plot_all_grads(results_dict, filename=None, div=False):
    plt.figure()
    axes = plt.gca()
    #axes.set_xlim([0, ARGS.k[-1]*100])
    axes.set_ylim([0, 0.5])
    axes.set_xlabel('% pixels removed')
    axes.set_ylabel('Absolute fractional output change')

    x_labels = [i*100 for i in ARGS.k]
    for key, v in results_dict.items():
        # Plot the mean and variance of the predictive distribution on the 100000 data points.
        plt.plot(ARGS.k, np.array(v[0]), linewidth=1.2, label=str(key))
        if div is False:
            plt.fill_between(ARGS.k, np.array(v[0]) - np.array(v[1]), np.array(v[0]) + np.array(v[1]), alpha=1/3)
    plt.xticks(ARGS.k, x_labels, rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(filename + ".png")
    # plt.show()


def initialize_grad_cam(model_name, device, pretrained=True):
    model = models.__dict__[model_name](pretrained=pretrained)
    model.to(device)
    model.eval()

    gcam = grad_cam.GradCAM(model=model)

    return model, gcam


def compute_saliency_per_grad(grad_type, grad, data):
    saliency = None

    # FULLGRAD
    if grad_type == "fullgrad":
        # print("calculating saliency")
        saliency = grad.saliency(data)

    elif grad_type == "gradcam":
        probs, ids = grad.forward(data)
        # Grad-CAM
        grad.backward(ids=ids[:, [0]])
        saliency = grad.generate(target_layer=ARGS.target_layer)

    return saliency


def get_filename(result_path, salient_type, grad_type, index):
    filename = result_path + ARGS.dataset + "/" + grad_type + "_" + ARGS.model + ARGS.model_type + \
               "_" + salient_type + "_" + str(index) + ".png"
    return filename


def save_saliency_map_batch(saliency, data, result_path, grad_type, salient_type, index):
    for i in range(len(data)):
        im = unnormalize(data[i, :, :, :].cpu())
        im = im.view(1, 3, 224, 224)[-1, :, :, :]
        reg = saliency[i, :, :, :]
        filename = get_filename(result_path, salient_type, grad_type, index)
        # print("filename:{}".format(filename))
        save_saliency_map(im, reg, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', default=[0.001, 0.005, 0.01, 0.05, 0.1], type=float,nargs="+",
                        help='Percentage of k% most salient pixels')
    parser.add_argument('--most_salient', default=False, type=bool,
                        help='most salient = True or False depending on retrain or pixel perturbation')
    parser.add_argument('--model', default="resnet", type=str,
                        help='which model to use')
    parser.add_argument('--model_type', default="18", type=str,
                        help='which model type: resnet_18, ...')
    parser.add_argument('--dataset', default="imagenet", type=str,
                        help='which dataset')
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
    parser.add_argument('--save_grad', default=False, type=bool,
                        help='saliency map to be saved?')
    parser.add_argument('--n_images', default=50, type=int,
                        help='Test for n_images images ')
    parser.add_argument('--n_save', default=50, type=int,
                        help='Save saliency maps for first n_save images')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Number of images passed through at once')

    ARGS = parser.parse_args()
    main()


# TODO 1) change to probs in abs_pergrad
# TODO 2) add KL divergence as metric
# TODO 3) run experiment vgg-16
# TODO 4) new metric add
# TODO 5) add inputgrad
# TODO 6) scrape google for doctor and nurse images
